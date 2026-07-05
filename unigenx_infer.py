# -*- coding: utf-8 -*-
import json
import os
from dataclasses import asdict

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, AllChem
from rdkit.Geometry import Point3D
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed
from transformers.generation.configuration_utils import GenerationConfig

from unigenx.data.dataset import MODE, UniGenXDataset, pad_1d_unsqueeze
from unigenx.data.tokenizer import UniGenXTokenizer
from unigenx.logging import logger
from unigenx.model.config import (
    UniGenXConfig,
    UniGenXInferenceConfig,
    UniGenXInferencedenovoConfig,
)
from unigenx.model.wrapper import UniGenX
from unigenx.utils import arg_utils
from unigenx.utils.checkpoint import load_checkpoint
from unigenx.utils.cli_utils import cli
from unigenx.utils.move_to_device import move_to_device

SPECIAL_TOKEN_IDS = {"bos": None, "eos": None, "padding": None, "coord": None}

JSON_SERIALIZABLE_TYPES = (np.float32,)


def convert_json_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@cli(UniGenXConfig, UniGenXInferenceConfig)
def main(args):
    # region initial config--------
    set_seed(args.seed)
    logger.info(f"Initializing with seed: {args.seed}")

    config = arg_utils.from_args(args, UniGenXConfig)
    inference_config = arg_utils.from_args(args, UniGenXInferenceConfig)

    if inference_config.input_file is None:
        inference_config = arg_utils.from_args(args, UniGenXInferencedenovoConfig)

    checkpoints_state = load_checkpoint(config.loadcheck_path)
    saved_args = checkpoints_state["args"]

    saved_config = arg_utils.from_args(saved_args, UniGenXConfig)
    saved_config.tokenizer = "num"
    saved_config.diff_steps = config.diff_steps
    saved_config.target = config.target
    ## modify for dpm solver ##
    saved_config.is_solver = config.is_solver
    saved_config.solver_order = config.solver_order
    saved_config.solver_type = config.solver_type

    for k, v in asdict(config).items():
        if not hasattr(saved_config, k):
            setattr(saved_config, k, getattr(config, k))
    saved_config.update(asdict(inference_config))
    # endregion ---------------

    # region initial model --------
    logger.info(f"Loading tokenizer from {args.dict_path}")
    tokenizer = UniGenXTokenizer.from_file(args.dict_path, saved_config)

    SPECIAL_TOKEN_IDS.update(
        {
            "bos": tokenizer.bos_idx,
            "eos": tokenizer.eos_idx,
            "padding": tokenizer.padding_idx,
            "coord": tokenizer.coord_idx,
        }
    )

    # Auto-detect the diffusion head width from the checkpoint so a 3-channel
    # (learn_sigma=False) head -- e.g. mol_qm9.pt -- loads with no user flag.
    # target_channels for xyz is 3; a 6-wide head means learned sigma.
    _ckpt_container = checkpoints_state
    if "model" in _ckpt_container:
        _ckpt_container = _ckpt_container["model"]
    elif "module" in _ckpt_container:
        _ckpt_container = _ckpt_container["module"]
    _diff_key = next(
        (
            k
            for k in _ckpt_container
            if k.endswith("diffloss.net.final_layer.linear.weight")
        ),
        None,
    )
    if _diff_key is not None:
        saved_config.learn_sigma = _ckpt_container[_diff_key].shape[0] == 3 * 2

    model = UniGenX(saved_config)
    model.eval()

    logger.info(f"Loading model from {args.loadcheck_path}")
    model.load_pretrained_weights(args.loadcheck_path)
    model.cuda()

    # region data&GenConfig ---------------
    logger.info(f"Loading inference data from {args.input_file}")
    saved_config.mask_token_id = tokenizer.mask_idx
    if inference_config.input_file is not None:
        gen_config = GenerationConfig(
            pad_token_id=SPECIAL_TOKEN_IDS["padding"],
            eos_token_id=SPECIAL_TOKEN_IDS["eos"],
            use_cache=True,
            max_length=saved_config.max_position_embeddings,
            return_dict_in_generate=True,
        )
        # Here we use sampling method to generate words.
        sample_config = GenerationConfig(
            pad_token_id=SPECIAL_TOKEN_IDS["padding"],
            eos_token_id=SPECIAL_TOKEN_IDS[
                "coord"
            ],  # Use coord_idx as the END OF SENTENCE token
            use_cache=True,
            max_length=saved_config.max_position_embeddings,
            return_dict_in_generate=True,
        )

        dataset = UniGenXDataset(
            tokenizer,
            args.input_file,
            saved_config,
            shuffle=False,
            mode=MODE.INFER,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.infer_batch_size,
            shuffle=False,
            collate_fn=dataset.collate,
            drop_last=False,
        )

        # endregion ---------------

        # region infer loop -------
        index = 0  # index of the currently processed data
        logger.info("Starting generation process...")
        with open(args.output_file, "w") as fw:
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    batch = move_to_device(batch, "cuda")

                    if args.target in {"material", "mol", "uni_mat", "uni_mol"}:
                        # Standard Gen
                        ret = model.net.generate(
                            input_ids=batch["input_ids"],
                            coordinates_mask=batch["coordinates_mask"],
                            generation_config=gen_config,
                            max_length=batch["coordinates_mask"].shape[1],
                        )
                        coordinates_mask = batch["coordinates_mask"]

                    elif args.target == "cond_mol":
                        # Two-phase conditional-molecule generation. Sequence
                        # layout: <bos> [<prop_i> propval_i]*num_cond <w> smiles
                        # <coord> [num_cond propval + n atom coords]. Supports one
                        # or many joint property constraints (num_cond) per mol.
                        # Phase 1: sample the SMILES sequence conditioned on the
                        # property prefix (stops at <coord>).
                        ret = model.net.generate(  # Phase1 result
                            input_ids=batch["input_ids"],
                            coordinates_mask=batch["coordinates_mask"],
                            generation_config=sample_config,
                            input_coordinates=batch["input_coordinates"],
                            only_seq=True,
                            max_length=dataset.max_position_embeddings // 2,
                            do_sample=True,
                            top_p=0.8,
                            temperature=0.6,
                        )

                        tokens = ret.sequences.cpu().numpy()
                        input_coordinates_batch = (
                            batch["input_coordinates"].cpu().numpy()
                        )
                        num_conds = batch["num_cond"].cpu().numpy()
                        batchsize = tokens.shape[0]
                        valid_tokens = []
                        atom_nums = []
                        skip_index_list = []
                        smiles = []
                        num_conds_valid = []
                        skip_index = index
                        input_coordinates_filtered = []
                        # input_coordinates are concatenated across the batch with
                        # num_cond rows per sample; walk them with a running offset.
                        begin_input_coord_idx = 0
                        for batch_idx in range(batchsize):
                            num_cond = int(num_conds[batch_idx])
                            sentence = []
                            if SPECIAL_TOKEN_IDS["coord"] in tokens[batch_idx]:
                                for token_idx in range(len(tokens[batch_idx])):
                                    if tokens[batch_idx][token_idx] not in [
                                        SPECIAL_TOKEN_IDS["bos"],
                                        SPECIAL_TOKEN_IDS["eos"],
                                        SPECIAL_TOKEN_IDS["padding"],
                                    ]:
                                        sentence.append(
                                            tokenizer.get_tok(
                                                tokens[batch_idx][token_idx]
                                            )
                                        )
                                    if (
                                        tokens[batch_idx][token_idx]
                                        == SPECIAL_TOKEN_IDS["coord"]
                                    ):
                                        break
                                # drop the [<prop_i> propval_i]*num_cond <w> prefix
                                # (2*num_cond + 1 tokens) and the trailing <coord>
                                mol = Chem.MolFromSmiles(
                                    "".join(sentence[2 * num_cond + 1 : -1]),
                                    sanitize=False,
                                )
                                if mol is not None:
                                    atom_nums.append(mol.GetNumAtoms())
                                    smiles.append(sentence[2 * num_cond + 1 : -1])
                                    num_conds_valid.append(num_cond)
                                    for k in range(num_cond):
                                        input_coordinates_filtered.append(
                                            input_coordinates_batch[
                                                begin_input_coord_idx + k
                                            ]
                                        )
                                    token = tokens[batch_idx]
                                    # Strip left padding: phase-1 output is LEFT-padded when a
                                    # batch mixes records of different prefix length, so anchor
                                    # the phase-2 slice at <bos> (not index 0) to keep the
                                    # freshly built coordinates_mask aligned with the real
                                    # tokens; otherwise the diffusion head is conditioned on
                                    # pad positions and the coordinates diverge.
                                    bos_pos = np.where(
                                        token == SPECIAL_TOKEN_IDS["bos"]
                                    )[0][0]
                                    coord_pos = np.where(
                                        token == SPECIAL_TOKEN_IDS["coord"]
                                    )[0][0]
                                    valid_tokens.append(
                                        token[bos_pos : coord_pos + 1].tolist()
                                    )
                                else:
                                    skip_index_list.append(skip_index)
                            else:
                                skip_index_list.append(skip_index)
                            skip_index += 1
                            begin_input_coord_idx += num_cond

                        if len(valid_tokens) == 0:
                            index += batchsize
                            continue
                        # prepare phase2 input: the num_cond propval slots stay
                        # mask=1 (input condition), the n atoms are generated.
                        origin_coordinates_mask = []
                        for token_seq, atom_count, num_cond in zip(
                            valid_tokens, atom_nums, num_conds_valid
                        ):
                            mask = (
                                [0]
                                + [elem for _ in range(num_cond) for elem in [0, 1]]
                                + [0] * (len(token_seq) - 2 * num_cond - 1)
                                + [1] * atom_count
                                + [0]
                            )
                            origin_coordinates_mask.append(mask)

                        max_tokens = max(len(token) for token in valid_tokens)
                        max_masks = max(
                            len(origin_coordinates_mask[i])
                            + max_tokens
                            - len(valid_tokens[i])
                            for i in range(len(origin_coordinates_mask))
                        )
                        input_ids = torch.cat(
                            [
                                pad_1d_unsqueeze(
                                    torch.Tensor(token).long(),
                                    max_tokens,
                                    max_tokens - len(token),
                                    SPECIAL_TOKEN_IDS["padding"],
                                )
                                for token in valid_tokens
                            ]
                        )
                        coordinates_mask = torch.cat(
                            [
                                pad_1d_unsqueeze(
                                    torch.Tensor(mask).long(),
                                    max_masks,
                                    max_tokens - len(token),
                                    SPECIAL_TOKEN_IDS["padding"],
                                )
                                for mask, token in zip(
                                    origin_coordinates_mask, valid_tokens
                                )
                            ]
                        )
                        input_coordinates_filtered = torch.cat(
                            [
                                torch.from_numpy(s).unsqueeze(0)
                                for s in input_coordinates_filtered
                            ]
                        ).to(torch.float32)
                        input_ids = move_to_device(input_ids, "cuda")
                        coordinates_mask = move_to_device(coordinates_mask, "cuda")
                        input_coordinates_filtered = move_to_device(
                            input_coordinates_filtered, "cuda"
                        )
                        # Phase2 Generation
                        ret = model.net.generate(
                            input_ids=input_ids,
                            coordinates_mask=coordinates_mask,
                            input_coordinates=input_coordinates_filtered,
                            do_sample=False,
                            generation_config=gen_config,
                            max_length=coordinates_mask.shape[1],
                        )

                    elif args.target == "cond_mat":
                        # Two-phase conditional-material generation. Sequence layout:
                        #   <prop> propval <bos> n*atoms <coord> [3 lattice + n atoms]
                        # Phase 1: sample the element sequence conditioned on the
                        # property prefix (stops at <coord>).
                        ret = model.net.generate(
                            input_ids=batch["input_ids"],
                            coordinates_mask=batch["coordinates_mask"],
                            generation_config=sample_config,
                            input_coordinates=batch["input_coordinates"],
                            only_seq=True,
                            max_length=dataset.max_position_embeddings // 2,
                            do_sample=True,
                            top_p=args.top_p if args.top_p is not None else 0.8,
                            temperature=(
                                args.temperature
                                if args.temperature is not None
                                else 0.6
                            ),
                        )
                        tokens = ret.sequences.cpu().numpy()
                        input_coordinates_batch = (
                            batch["input_coordinates"].cpu().numpy()
                        )
                        bs = tokens.shape[0]
                        atom_nums = []
                        skip_index_list = []
                        skip_index = index
                        input_coordinates_filtered = []
                        valid_tokens = []
                        assert bs == batch["input_coordinates"].shape[0]

                        for batch_idx in range(bs):
                            # Require the first special token after the
                            # <prop> propval <bos> prefix to be <coord>.
                            flag = True
                            _j = 3
                            while _j < len(tokens[batch_idx]):
                                tok = tokenizer.get_tok(tokens[batch_idx][_j])
                                if tok == "<coord>":
                                    break
                                if tok[0] == "<":
                                    flag = False
                                    break
                                _j += 1
                            sent = []
                            if SPECIAL_TOKEN_IDS["coord"] in tokens[batch_idx] and flag:
                                for token_idx in range(len(tokens[batch_idx])):
                                    if tokens[batch_idx][token_idx] not in [
                                        SPECIAL_TOKEN_IDS["bos"],
                                        SPECIAL_TOKEN_IDS["eos"],
                                        SPECIAL_TOKEN_IDS["padding"],
                                    ]:
                                        sent.append(
                                            tokenizer.get_tok(
                                                tokens[batch_idx][token_idx]
                                            )
                                        )
                                    if (
                                        tokens[batch_idx][token_idx]
                                        == SPECIAL_TOKEN_IDS["coord"]
                                    ):
                                        break
                                # sent = [<prop> propval elem... <coord>]:
                                # ignore <prop>, propval, <coord> when counting atoms
                                atom_nums.append(len(sent) - 3)
                                dataset.data[skip_index]["sites"] = sent[2:-1]
                                input_coordinates_filtered.append(
                                    input_coordinates_batch[batch_idx]
                                )
                                token = tokens[batch_idx]
                                # Strip left padding (see the cond_mol phase-2 slice above):
                                # anchor at <bos> so a mixed-length batch keeps the
                                # coordinates_mask aligned with the real tokens.
                                bos_pos = np.where(token == SPECIAL_TOKEN_IDS["bos"])[
                                    0
                                ][0]
                                coord_pos = np.where(
                                    token == SPECIAL_TOKEN_IDS["coord"]
                                )[0][0]
                                valid_tokens.append(
                                    token[bos_pos : coord_pos + 1].tolist()
                                )
                            else:
                                skip_index_list.append(skip_index)
                            skip_index += 1

                        if len(valid_tokens) == 0:
                            index += bs
                            continue

                        # Phase 2 mask: propval slot stays mask=1 (input
                        # condition); the generation region is 3 lattice + n atoms.
                        origin_coordinates_mask = [
                            [0, 1, 0]  # <prop> propval <bos>
                            + [0] * (atom_nums[k] + 1)  # n atoms + <coord>
                            + [1] * (atom_nums[k] + 3)  # 3 lattice + n atoms
                            + [0]
                            for k in range(len(valid_tokens))
                        ]
                        max_tokens = max(len(t) for t in valid_tokens)
                        max_masks = max(
                            len(origin_coordinates_mask[k])
                            + max_tokens
                            - len(valid_tokens[k])
                            for k in range(len(origin_coordinates_mask))
                        )
                        input_ids = torch.cat(
                            [
                                pad_1d_unsqueeze(
                                    torch.Tensor(token).long(),
                                    max_tokens,
                                    max_tokens - len(token),
                                    SPECIAL_TOKEN_IDS["padding"],
                                )
                                for token in valid_tokens
                            ]
                        )
                        coordinates_mask = torch.cat(
                            [
                                pad_1d_unsqueeze(
                                    torch.Tensor(mask).long(),
                                    max_masks,
                                    max_tokens - len(token),
                                    SPECIAL_TOKEN_IDS["padding"],
                                )
                                for mask, token in zip(
                                    origin_coordinates_mask, valid_tokens
                                )
                            ]
                        )
                        input_coordinates_filtered = torch.cat(
                            [
                                torch.from_numpy(s).unsqueeze(0)
                                for s in input_coordinates_filtered
                            ]
                        ).to(torch.float32)
                        input_ids = move_to_device(input_ids, "cuda")
                        coordinates_mask = move_to_device(coordinates_mask, "cuda")
                        input_coordinates_filtered = move_to_device(
                            input_coordinates_filtered, "cuda"
                        )
                        # Phase 2: generate lattice + atom coordinates.
                        ret = model.net.generate(
                            input_ids=input_ids,
                            coordinates_mask=coordinates_mask,
                            input_coordinates=input_coordinates_filtered,
                            do_sample=False,
                            generation_config=gen_config,
                            max_length=coordinates_mask.shape[1],
                        )

                    elif args.target == "prot":
                        # Protein-backbone (Cα) conformation generation for the
                        # MD / fast-folding-protein setup. Mirrors the standalone
                        # gen_threedimargendiff_prot.py: for each input sequence
                        # sample ``num_topk`` conformations. Short sequences
                        # (<=256 residues) are generated in one pass; longer ones
                        # use a sliding window over the residue sequence. This is
                        # the baseline path (sequence tokens + coordinate mask);
                        # precomputed ESM-2 embeddings, when used, are an external
                        # conditioning stream and not required here. The two-phase
                        # top-k / index bookkeeping assumes one protein per batch,
                        # i.e. ``--infer_batch_size 1`` (as in the source script).
                        # This branch is fully self-contained and ``continue``s
                        # past the shared result-processing below, so it never
                        # affects the material/mol/cond_* targets.
                        num_topk = 5
                        if batch["input_ids"].shape[1] - 2 <= 256:
                            input_ids = batch["input_ids"].repeat(num_topk, 1)
                            coordinates_mask = batch["coordinates_mask"].repeat(
                                num_topk, 1
                            )
                            ret = model.net.generate(
                                input_ids=input_ids,
                                coordinates_mask=coordinates_mask,
                                generation_config=gen_config,
                                max_length=coordinates_mask.shape[1],
                            )
                            decoded_results = tokenizer.decode_batch(
                                ret.sequences.cpu().numpy(),
                                ret.coordinates.cpu().numpy(),
                                coordinates_mask.cpu().numpy(),
                                args.target,
                            )
                            record = dataset.get_infer_record_prot(index)
                            for sent, atom_coordinates in decoded_results:
                                if args.verbose:
                                    print(f"Generated protein: {sent}")
                                if "prediction" not in record:
                                    record["prediction"] = {
                                        "coordinates": [atom_coordinates],
                                    }
                                else:
                                    record["prediction"]["coordinates"].append(
                                        atom_coordinates
                                    )
                            fw.write(
                                json.dumps(
                                    record,
                                    default=convert_json_serializable,
                                )
                                + "\n"
                            )
                            index += 1
                        else:
                            # Sliding window for long sequences (>256 residues):
                            # generate the first 256-residue block, then advance by
                            # ``window_width`` residues, re-conditioning on the
                            # overlap, and stitch the coordinate blocks together.
                            seq = batch["input_ids"][:, 1:-1].cpu()
                            bos = tokenizer.bos_idx
                            coord_idx = tokenizer.coord_idx
                            start = 0
                            window_width = 128
                            coordinates = None
                            while True:
                                if start == 0:
                                    input_ids = torch.cat(
                                        [
                                            torch.full((1, 1), bos),
                                            seq[:, start : start + 256],
                                            torch.full((1, 1), coord_idx),
                                        ],
                                        dim=1,
                                    )
                                    coordinates_mask = torch.tensor(
                                        [0 for _ in range(input_ids.shape[1])]
                                        + [1 for _ in range((input_ids.shape[1] - 2))]
                                        + [0]
                                    ).unsqueeze(0)
                                else:
                                    input_ids = torch.cat(
                                        [
                                            torch.full((1, 1), bos),
                                            seq[:, start : start + 256],
                                            torch.full((1, 1), coord_idx),
                                            torch.full(
                                                (1, 256 - window_width),
                                                tokenizer.mask_idx,
                                            ),
                                        ],
                                        dim=1,
                                    )
                                    coordinates_mask = torch.tensor(
                                        [
                                            0
                                            for _ in range(
                                                input_ids.shape[1]
                                                - (256 - window_width)
                                            )
                                        ]
                                        + [
                                            1
                                            for _ in range(
                                                input_ids.shape[1]
                                                - 2
                                                - (256 - window_width)
                                            )
                                        ]
                                        + [0]
                                    ).unsqueeze(0)

                                input_ids = input_ids.repeat(num_topk, 1)
                                coordinates_mask = coordinates_mask.repeat(num_topk, 1)
                                input_ids = move_to_device(input_ids, "cuda")
                                coordinates_mask = move_to_device(
                                    coordinates_mask, "cuda"
                                )
                                ret = model.net.generate(
                                    input_ids=input_ids,
                                    coordinates_mask=coordinates_mask,
                                    generation_config=gen_config,
                                    max_length=coordinates_mask.shape[1],
                                )
                                top5_coordinates = ret.coordinates.view(
                                    num_topk, -1, ret.coordinates.shape[-1]
                                )
                                if start == 0:
                                    coordinates = top5_coordinates
                                else:
                                    coordinates = torch.cat(
                                        [
                                            coordinates,
                                            top5_coordinates[
                                                :, 256 - window_width :, :
                                            ],
                                        ],
                                        dim=1,
                                    )
                                start += window_width
                                if coordinates.shape[1] == seq.shape[1]:
                                    record = dataset.get_infer_record_prot(index)
                                    record["prediction"] = {
                                        "coordinates": coordinates.tolist(),
                                    }
                                    fw.write(
                                        json.dumps(
                                            record,
                                            default=convert_json_serializable,
                                        )
                                        + "\n"
                                    )
                                    index += 1
                                    break
                        continue

                    elif args.target in {"dock", "misato"}:
                        # Protein-ligand docking. The pocket coordinates are
                        # given via input_coordinates (dock: protein pocket,
                        # mask==2; misato: apo + holo pocket, mask==2/1) and the
                        # model generates the ligand pose (dock) / the docked
                        # ligand (misato). decode_batch returns
                        # (sent, lattice, atom_coordinates): lattice is the given
                        # pocket block, atom_coordinates the generated ligand
                        # (dock) or the holo-pocket + ligand stream (misato).
                        # One output record per input structure; this branch is
                        # self-contained and continues past the shared
                        # result-processing below.
                        ret = model.net.generate(
                            input_ids=batch["input_ids"],
                            coordinates_mask=batch["coordinates_mask"],
                            generation_config=gen_config,
                            input_coordinates=batch["input_coordinates"],
                            max_length=batch["coordinates_mask"].shape[1],
                        )
                        decoded_results = tokenizer.decode_batch(
                            ret.sequences.cpu().numpy(),
                            ret.coordinates.cpu().numpy(),
                            batch["coordinates_mask"].cpu().numpy(),
                            args.target,
                        )
                        ids = batch["id"].cpu().numpy()
                        gt_coords = (
                            batch["gt_coords"].cpu().numpy()
                            if args.target == "dock"
                            else None
                        )
                        _lig_offset = 0
                        for i in range(len(decoded_results)):
                            sent, lattice, atom_coordinates = decoded_results[i]
                            if args.verbose:
                                print(f"Generated docking: {sent}")
                            if args.target == "dock":
                                item = dataset.get_simple_item_docking(ids[i])
                                _j = _lig_offset + len(atom_coordinates)
                                item["prediction"] = {
                                    "prot_coords": lattice,
                                    "ligand_coords": atom_coordinates,
                                }
                                item["ligand_gt"] = gt_coords[_lig_offset:_j]
                                _lig_offset = _j
                            else:  # misato
                                item = dataset.get_simple_item_misato(ids[i])
                                apo_coords = item["apo_coords"]
                                holo_coords = atom_coordinates[: len(apo_coords)]
                                ligand_coords = atom_coordinates[len(apo_coords) :]
                                item["prediction"] = {
                                    "apo_coords": lattice,
                                    "holo_coords": holo_coords,
                                    "ligand_coords": ligand_coords,
                                }
                            fw.write(
                                json.dumps(item, default=convert_json_serializable)
                                + "\n"
                            )
                            index += 1
                        continue

                    elif args.target == "ecnum":
                        # EC-number conditioned enzyme (protein-sequence) design.
                        # The prompt is <bos> <ec1> L1 <ec2> L2 <ec3> L3 <prot>
                        # (built in get_infer_item_ecnum); the model samples the
                        # amino-acid sequence and stops at <coord> (sample_config
                        # uses coord_idx as EOS). Sequence-only: a fresh all-zeros
                        # coordinates_mask spanning the full generation length is
                        # passed so no coordinate slot is ever produced (mirrors
                        # gen_threedimargendiff_ecnum.py). Self-contained branch:
                        # continues past the shared coordinate result-processing.
                        gen_len = saved_config.max_position_embeddings
                        coordinates_mask = torch.zeros(
                            (batch["input_ids"].shape[0], gen_len), dtype=torch.long
                        )
                        coordinates_mask = move_to_device(coordinates_mask, "cuda")
                        sample_kwargs = {"do_sample": True}
                        if args.top_p is not None:
                            sample_kwargs["top_p"] = args.top_p
                        if args.temperature is not None:
                            sample_kwargs["temperature"] = args.temperature
                        ret = model.net.generate(
                            input_ids=batch["input_ids"],
                            coordinates_mask=coordinates_mask,
                            generation_config=sample_config,
                            max_length=coordinates_mask.shape[1],
                            **sample_kwargs,
                        )
                        tokens = ret.sequences.cpu().numpy()
                        valid_aa = set("ARNDCQEGHILKMFPSTWYV")
                        for i in range(tokens.shape[0]):
                            sent = []
                            if SPECIAL_TOKEN_IDS["coord"] in tokens[i]:
                                for j in range(len(tokens[i])):
                                    if tokens[i][j] == SPECIAL_TOKEN_IDS["coord"]:
                                        break
                                    if tokens[i][j] not in (
                                        SPECIAL_TOKEN_IDS["bos"],
                                        SPECIAL_TOKEN_IDS["eos"],
                                        SPECIAL_TOKEN_IDS["padding"],
                                    ):
                                        sent.append(tokenizer.get_tok(tokens[i][j]))
                            sent = "".join(sent)
                            # split off the EC prefix; keep the amino-acid sequence
                            parts = sent.split("<prot>")
                            aa_seq = None
                            if len(parts) == 2:
                                candidate = parts[-1]
                                # only keep sequences of standard amino acids
                                if not [c for c in candidate if c not in valid_aa]:
                                    aa_seq = candidate
                            record = dataset.data[index]
                            if args.verbose:
                                print(
                                    f"Generated enzyme (EC "
                                    f"{record.get('EC_number')}): {aa_seq}"
                                )
                            record["prediction"] = {"seq": aa_seq}
                            fw.write(
                                json.dumps(record, default=convert_json_serializable)
                                + "\n"
                            )
                            index += 1
                        continue

                    # region Result processing -------
                    decoded_results = tokenizer.decode_batch(
                        ret.sequences.cpu().numpy(),
                        ret.coordinates.cpu().numpy(),
                        coordinates_mask.cpu().numpy(),
                        args.target,
                    )

                    for i in range(len(decoded_results)):
                        if args.target == "material" or args.target == "uni_mat":
                            sentences, lattice, atom_coordinates = decoded_results[i]
                            if args.verbose:
                                print(f"Generated material:{sentences}")
                            dataset.data[index]["prediction"] = {
                                "lattice": lattice,
                                "coordinates": atom_coordinates,
                            }
                            fw.write(
                                json.dumps(
                                    dataset.data[index],
                                    default=convert_json_serializable,
                                )
                                + "\n"
                            )
                            index += 1
                        elif args.target == "mol" or args.target == "uni_mol":
                            sentences, atom_coordinates = decoded_results[i]
                            if args.verbose:
                                print(f"Generated molecule: {sentences}")
                            dataset.data[index]["prediction"] = {
                                "coordinates": atom_coordinates,
                            }
                            fw.write(
                                json.dumps(
                                    dataset.data[index],
                                    default=convert_json_serializable,
                                )
                                + "\n"
                            )
                            index += 1
                        elif args.target == "cond_mat":
                            # 4-tuple: cond_val is the model's reconstruction of
                            # the conditioning value (coordinate slot 0).
                            (
                                sentences,
                                cond_val,
                                lattice,
                                atom_coordinates,
                            ) = decoded_results[i]
                            if args.verbose:
                                print(f"Conditional material: {sentences}")
                            while index in skip_index_list:
                                index += 1
                            dataset.data[index]["prediction"] = {
                                "lattice": lattice,
                                "coordinates": atom_coordinates,
                                "cond_val": cond_val,
                            }
                            fw.write(
                                json.dumps(
                                    dataset.data[index],
                                    default=convert_json_serializable,
                                )
                                + "\n"
                            )
                            index += 1
                        elif args.target == "cond_mol":
                            sentences, atom_coordinates = decoded_results[i]
                            if args.verbose:
                                print(f"Conditional generated: {sentences}")
                            while index in skip_index_list:
                                index += 1
                            ans = dict()
                            # decoded coords are [num_cond propval rows, n atoms];
                            # drop the leading property-value rows to keep atoms.
                            ans["coordinates"] = atom_coordinates[num_conds_valid[i] :]
                            ans["smi"] = "".join(smiles[i])
                            ans["prop"] = dataset.data[index]["prop"]
                            ans["prop_val"] = dataset.data[index]["prop_val"]
                            fw.write(
                                json.dumps(ans, default=convert_json_serializable)
                                + "\n"
                            )
                            index += 1
                    # endregion ---------------
        # endregion ---------------
    else:
        # region denovo case -------------
        # if inference_config.sample:
        num_batches = inference_config.sample_size // inference_config.infer_batch_size
        batches = []
        for _ in range(num_batches):
            input_ids = torch.full(
                (inference_config.infer_batch_size, 1), tokenizer.bos_idx
            )
            coordinates_mask = torch.zeros(
                (inference_config.infer_batch_size, inference_config.sample_max_length)
            )
            batches.append(
                {"input_ids": input_ids, "coordinates_mask": coordinates_mask}
            )

        if inference_config.sample_size % inference_config.infer_batch_size != 0:
            remainder_size = (
                inference_config.sample_size % inference_config.infer_batch_size
            )
            input_ids = torch.full((remainder_size, 1), tokenizer.bos_idx)
            coordinates_mask = torch.zeros(
                (remainder_size, inference_config.sample_max_length)
            )
            batches.append(
                {"input_ids": input_ids, "coordinates_mask": coordinates_mask}
            )

        gen_config = GenerationConfig(
            pad_token_id=tokenizer.padding_idx,
            eos_token_id=tokenizer.coord_idx,
            use_cache=True,
            max_length=saved_config.max_position_embeddings,
            return_dict_in_generate=True,
        )
        os.makedirs(inference_config.output_file, exist_ok=True)
        with open(os.path.join(inference_config.output_file, "sample.txt"), "w") as fw:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(batches)):
                    batch = move_to_device(batch, "cuda")
                    ret = model.net.generate(
                        input_ids=batch["input_ids"],
                        coordinates_mask=batch["coordinates_mask"],
                        generation_config=gen_config,
                        max_length=batch["coordinates_mask"].shape[1],
                        do_sample=True,
                        top_p=inference_config.top_p,
                        temperature=inference_config.temperature,
                    )
                    tokens = ret.sequences.cpu().numpy()
                    ret = []
                    bs = tokens.shape[0]
                    for i in range(bs):
                        sent = []
                        if tokenizer.coord_idx in tokens[i]:
                            for j in range(len(tokens[i])):
                                if tokens[i][j] not in [
                                    tokenizer.bos_idx,
                                    tokenizer.eos_idx,
                                    tokenizer.padding_idx,
                                ]:
                                    sent.append(tokenizer.get_tok(tokens[i][j]))
                                if tokens[i][j] == tokenizer.coord_idx:
                                    break
                            ret.append(sent)

                    for i in range(len(ret)):
                        sent = ret[i]
                        fw.write(" ".join(sent) + "\n")

                    tokens = [
                        token[
                            : np.where(token == tokenizer.coord_idx)[0][0] + 1
                        ].tolist()
                        for token in tokens
                        if tokenizer.coord_idx in token
                    ]
                    tokens = [token for token in tokens if len(token) <= 22]
                    origin_coordinates_mask = [
                        [0 for _ in range(len(token))]
                        + [1 for _ in range((len(token) + 1))]
                        + [0]
                        for token in tokens
                    ]
                    max_tokens = max(len(token) for token in tokens)
                    max_masks = max(len(mask) for mask in origin_coordinates_mask)
                    input_ids = torch.cat(
                        [
                            pad_1d_unsqueeze(
                                torch.Tensor(token).long(),
                                max_tokens,
                                max_tokens - len(token),
                                tokenizer.padding_idx,
                            )
                            for token in tokens
                        ]
                    )
                    coordinates_mask = torch.cat(
                        [
                            pad_1d_unsqueeze(
                                torch.Tensor(mask).long(),
                                max_masks,
                                max_tokens - len(token),
                                tokenizer.padding_idx,
                            )
                            for mask, token in zip(origin_coordinates_mask, tokens)
                        ]
                    )

                    input_ids = move_to_device(input_ids, "cuda")
                    coordinates_mask = move_to_device(coordinates_mask, "cuda")
                    gen_config = GenerationConfig(
                        pad_token_id=tokenizer.padding_idx,
                        eos_token_id=tokenizer.eos_idx,
                        use_cache=True,
                        max_length=saved_config.max_position_embeddings,
                        return_dict_in_generate=True,
                    )
                    ret = model.net.generate(
                        input_ids=input_ids,
                        coordinates_mask=coordinates_mask,
                        do_sample=False,
                        generation_config=gen_config,
                        max_length=coordinates_mask.shape[1],
                    )
                    sentences = ret.sequences.cpu().numpy()
                    coordinates = ret.coordinates.cpu().numpy()
                    masks = coordinates_mask.cpu().numpy()
                    ret2 = tokenizer.decode_batch(sentences, coordinates, masks)

                    for i in range(len(ret2)):
                        sent, lattice, atom_coordinates = ret2[i]
                        if inference_config.verbose:
                            print(sent)
                        species = sent.split("<coord>")[0].split()
                        print(species)
                        if len(atom_coordinates) > len(species):
                            atom_coordinates = atom_coordinates[: len(species)]
                        try:
                            structure = Structure(
                                lattice=lattice,
                                species=species,
                                coords=atom_coordinates,
                            )
                            cif = CifWriter(structure)
                            cif.write_file(
                                f"{inference_config.output_file}/gen_{batch_idx * inference_config.infer_batch_size + i}.cif"
                            )
                        except:
                            print("fail")
            # endregion ---------------


if __name__ == "__main__":
    main()
