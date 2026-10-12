"""content fingerprints of processed tables, independent of how the Parquet file is encoded.

a table digest is a sha256 over the values of every column of a schema in row order. a fingerprint combines the
digests of the consecutive blocks of a table, one per capture or per year, so a table built again from the same raw
files can be checked block by block against the manifest of a processed version.
"""

import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa


def table_digest(frame: pd.DataFrame, schema: pa.Schema) -> str:
    # times count in whole seconds, categories by their names and numbers by their values, so neither the time unit nor
    # the category codes nor the integer width change the digest
    h = hashlib.sha256()
    for field in schema:
        col = frame[field.name]
        if pa.types.is_timestamp(field.type):
            data = col.to_numpy().astype("datetime64[s]").astype("<i8").tobytes()
        elif pa.types.is_dictionary(field.type):
            col = col.astype("category")
            names, codes = np.asarray(col.cat.categories.astype(str), dtype=object), col.cat.codes.to_numpy()
            if (codes < 0).any():
                raise ValueError(f"{field.name} holds missing values")
            data = ("\n".join(names[codes]) + "\n").encode()
        elif pa.types.is_floating(field.type):
            data = col.to_numpy(dtype="<f8").tobytes()
        else:
            data = col.to_numpy(dtype="<i8").tobytes()
        h.update(field.name.encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest()


def combine(blocks) -> str:
    # blocks are pairs of a block name and its digest, in row order
    h = hashlib.sha256()
    for name, digest in blocks:
        h.update(f"{name} {digest}\n".encode())
    return h.hexdigest()


def fingerprint(frame: pd.DataFrame, by: str, schema: pa.Schema) -> str:
    # one digest per run of equal values of the column by, in row order
    values = frame[by].astype("category")
    names, codes = values.cat.categories, values.cat.codes.to_numpy()
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    return combine((str(names[codes[s]]), table_digest(frame.iloc[s:e], schema)) for s, e in zip(starts, ends))
