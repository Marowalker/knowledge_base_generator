import csv
from constants import *
import re
from collections import defaultdict
import itertools
import time


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.job = None

    def start(self, job):
        if job is None:
            return None
        self.start_time = time.time()
        self.job = job
        print("[INFO] {job} started.".format(job=self.job))

    def stop(self):
        if self.job is None:
            return None
        elapsed_time = time.time() - self.start_time
        print("[INFO] {job} finished in {elapsed_time:0.3f} s."
              .format(job=self.job, elapsed_time=elapsed_time))
        self.job = None


class Log:
    verbose = True

    @staticmethod
    def log(text):
        if Log.verbose:
            print(text)


def clean_disease(disease):
    return disease[5:]


def search_sharp(row):
    for token in row:
        if "#" in token:
            return True
    return False


def make_chemicals():
    chemicals = []
    file = open(CTD)
    ctd = csv.reader(file)
    # ctd chemicals
    for row in ctd:
        if not search_sharp(row):
            chemical_id = tuple([row[0], row[1]])
            chemicals.append(chemical_id)
    file.close()

    # cdr chemicals
    for dataset in cdr_datasets:
        with open(CDR + dataset) as f:
            lines = f.readlines()
            regex = re.compile(r'^(\d+)\t(\d+)\t(\d+)\t([^\t]+)\t(\S+)\t(\S+)', re.U | re.I)
            for line in lines:
                matched = regex.search(line)
                if matched:
                    entity_line = matched.groups()
                    if entity_line[4] == 'Chemical':
                        chemical_ids = entity_line[5].split('t')
                        for idx in chemical_ids:
                            chemical_id = tuple([entity_line[3], idx])
                            chemicals.append(chemical_id)
            f.close()

    # clean chemical list and write to file
    chemicals_set = set(chemicals)
    print('Total chemicals:', len(chemicals_set))
    with open(ENTITY_PATH + "chemical2id.txt", "w") as f:
        f.write(str(len(chemicals_set)))
        f.write("\n")
        for pair in chemicals_set:
            f.write("{}\t{}".format(pair[0], pair[1]))
            f.write("\n")


def make_diseases():
    diseases = []
    file = open(CTD)
    ctd = csv.reader(file)
    # ctd diseases
    for row in ctd:
        if not search_sharp(row):
            disease_id = tuple([row[3], clean_disease(row[4])])
            diseases.append(disease_id)
    file.close()

    # cdr diseases
    for dataset in cdr_datasets:
        with open(CDR + dataset) as f:
            lines = f.readlines()
            regex = re.compile(r'^(\d+)\t(\d+)\t(\d+)\t([^\t]+)\t(\S+)\t(\S+)', re.U | re.I)
            for line in lines:
                matched = regex.search(line)
                if matched:
                    entity_line = matched.groups()
                    if entity_line[4] == 'Disease':
                        disease_ids = entity_line[5].split('|')
                        for idx in disease_ids:
                            disease_id = tuple([entity_line[3], idx])
                            diseases.append(disease_id)
            f.close()

    # clean disease list and write to file
    diseases_set = set(diseases)
    print('Total diseases:', len(diseases_set))
    with open(ENTITY_PATH + "disease2id.txt", "w") as f:
        f.write(str(len(diseases_set)))
        f.write("\n")
        for pair in diseases_set:
            f.write("{}\t{}".format(pair[0], pair[1]))
            f.write("\n")


def make_relations():
    print('Total relations:', len(list_relations))
    with open(ENTITY_PATH + "relation2id.txt", "w") as f:
        f.write(str(len(list_relations)))
        f.write("\n")
        for rel in list_relations:
            f.write("{}\t{}".format(rel, str(list_relations.index(rel))))
            f.write("\n")


def make_triples():
    triples = []
    ctd_chem_disease = []
    file = open(CTD)
    ctd = csv.reader(file)
    # ctd triples
    print("Making CTD triples...")
    for row in ctd:
        if not search_sharp(row):
            chem_disease = tuple([row[1], clean_disease(row[4])])
            ctd_chem_disease.append(chem_disease)
            if row[5] not in list_relations:
                triple = tuple([row[1], clean_disease(row[4]), str(list_relations.index("other"))])
            else:
                triple = tuple([row[1], clean_disease(row[4]), str(list_relations.index(row[5]))])
            triples.append(triple)
    file.close()

    # cdr triples
    print("Making CDR triples...")
    chemicals = defaultdict(list)
    diseases = defaultdict(list)
    relations = defaultdict(list)
    abstracts = []
    for dataset in cdr_datasets:
        with open(CDR + dataset) as f:
            lines = f.readlines()
            regex_ent = re.compile(r'^(\d+)\t(\d+)\t(\d+)\t([^\t]+)\t(\S+)\t(\S+)', re.U | re.I)
            regex_rel = re.compile(r'^([\d]+)\t(CID)\t([\S]+)\t([\S]+)$', re.U | re.I)
            for line in lines:
                matched = regex_ent.search(line)
                if matched:
                    entity_line = matched.groups()
                    if entity_line[4] == 'Chemical':
                        chem_id = entity_line[5].split('|')
                        chemicals[entity_line[0]] += chem_id
                    if entity_line[4] == 'Disease':
                        dis_id = entity_line[5].split('|')
                        diseases[entity_line[0]] += dis_id
                rel_matched = regex_rel.search(line)
                if rel_matched:
                    rel_line = rel_matched.groups()
                    abstracts.append(rel_line[0])
                    relations[rel_line[0]].append(tuple([rel_line[2], rel_line[3]]))

    print("Adding new CDR triples to list...")
    for abstract in abstracts:
        print("Processing abstract", abstract)
        chems = chemicals[abstract]
        dises = diseases[abstract]
        rels = relations[abstract]

        for (chem_id, dis_id) in rels:
            if (chem_id, dis_id) not in ctd_chem_disease:
                triple = tuple([chem_id, dis_id, list_relations.index('marker/mechanism')])
                # else:
                #     triple = tuple([chem_id, dis_id, list_relations.index('other')])
                triples.append(triple)

    triples_set = set(triples)
    print('Total triples:', len(triples_set))
    # print(len(triples_set))
    with open(ENTITY_PATH + "triple2id.txt", "w") as f:
        f.write(str(len(triples_set)))
        f.write("\n")
        for triple in triples_set:
            f.write("{}\t{}\t{}".format(triple[0], triple[1], triple[2]))
            f.write("\n")


def get_train_files():
    with open(ENTITY_PATH + "triple2id.txt") as f:
        lines = f.readlines()
        val_ratio = 0.85
        train_ratio = 0.7
        train_sample = int(len(lines) * train_ratio)
        val_sample = int(len(lines) * val_ratio)
        train_lines = lines[1:train_sample]
        val_lines = lines[train_sample:val_sample]
        test_lines = lines[val_sample:]
        f.close()
    with open(ENTITY_PATH + "train2id.txt", 'w') as f:
        f.write(str(len(train_lines)))
        f.write('\n')
        for line in train_lines:
            f.write(line)
        f.close()
    with open(ENTITY_PATH + "valid2id.txt", 'w') as f:
        f.write(str(len(val_lines)))
        f.write('\n')
        for line in val_lines:
            f.write(line)
        f.close()
    with open(ENTITY_PATH + "test2id.txt", 'w') as f:
        f.write(str(len(test_lines)))
        f.write('\n')
        for line in test_lines:
            f.write(line)
        f.close()


def count_vocab(filename):
    file = open(filename)
    lines = file.readlines()
    return int(lines[0].strip())


def count_wordnet(filename):
    file = open(filename)
    lines = file.readlines()
    return len(lines)
