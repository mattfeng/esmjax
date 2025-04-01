#!/usr/bin/env python

from pathlib import Path
import os

import xmlschema
import json

import tqdm

DATASET_DIR = Path("/home/gridsan/mattfeng/datasets/uniprot-2021_04/uniref90")

# Load the schema from your XSD file.
schema = xmlschema.XMLSchema(DATASET_DIR / "uniref.xsd")

print("Loaded schema.")

def process_entry(entry):
    data = schema.to_dict(entry)

    ur90_id = data["@id"]
    cluster_name = data["name"].split(": ")[1]

    go_functions = []

    for prop in data["property"]:
        if prop["@type"] == "GO Molecular Function":
            go_functions.append(prop["@value"])

    return {
        "ur90_id": ur90_id,
        "cluster_name": cluster_name,
        "go_functions":go_functions
    }

uniref_xml_path = DATASET_DIR / "uniref90.xml.noparent"

with open(uniref_xml_path, "r") as f, \
    open(DATASET_DIR / "uniref90_descriptions.jsonl", "w") as fout:

    entry = []
    add = False

    with tqdm.tqdm(total=os.path.getsize(uniref_xml_path), unit="B", unit_scale=True) as pbar:

        for line in f:
            pbar.update(len(line))

            line = line.strip()

            if line.startswith("<entry"):
                add = True
            
            if add:
                entry.append(line)

            if line.startswith("</entry>"):
                add = False
                data = process_entry("\n".join(entry))

                fout.write(json.dumps(data) + "\n")
                fout.flush()

                entry = []

        