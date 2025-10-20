from typing import Union, List, Optional

import pandas as pd
from pandas import DataFrame, concat
from .core_methods import SCFind
import h5py
import os
import re
from gensim.models import KeyedVectors
from collections import defaultdict, Counter
import numpy as np
from rapidfuzz import process, fuzz
import math
from string import ascii_letters


def read_w2v(path: str) -> KeyedVectors:
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    # Load the model as a binary format
    return model


def read_h5(key, value):
    if isinstance(value, h5py.Dataset):
        data = value[()]
        if data.dtype.names:
            data_ = {name: np.char.decode(data[name], 'utf-8') for name in data.dtype.names}
        else:
            data_ = {key: np.char.decode(data, 'utf-8')}
        data_pd = pd.DataFrame(data_)
        return data_pd
    if isinstance(value, h5py.Group):
        sub_dict = {}
        for sub_key, sub_value in value.items():
            sub_dict[sub_key] = read_h5(sub_key, sub_value)

        return sub_dict


def read_dictionaries(paths: Union[str, List[str]]) -> dict:
    dictionary = {}
    paths = [paths] if isinstance(paths, str) else paths
    for i in paths:
        if i.lower().endswith(".h5"):
            with h5py.File(i, 'r') as f:
                for key in f.keys():
                    data = read_h5(key, f[key])
                    dictionary.update(data)
                    # # if key already exists, append it to current values
                    # if key in dictionary:
                    #     for inner_key, values in dictionary[key].items():
                    #         dictionary[key][inner_key] = values + dict_data.get(inner_key, [])
                    #
                    # else:
                    #     dictionary[key] = dict_data
        else:
            name = re.sub(r".*/|\..*", "", i)
            name = re.sub(r"_.*", "", name)
            # read non .h5 file and add it as a list in dictionary
            with open(i, 'r') as file:
                content = file.read()
                # if key name already exist, append it to current values
                if name in dictionary:
                    for inner_key, values in dictionary[name].items():
                        dictionary[name][inner_key] = values + [content]
                else:
                    dictionary[name] = [content]

    return dictionary


def read_all_dictionaries(w2v_path: str,
                          dictionary_paths: Union[str, List[str]],
                          priority: Optional[Union[str, List[str]]] = None
                          ) -> dict:
    if not dictionary_paths:
        raise ValueError("Please provide paths to dictionaries.")
    dictionary_paths = [dictionary_paths] if isinstance(dictionary_paths, str) else dictionary_paths

    dictionaries = read_dictionaries(dictionary_paths)

    default = ["meshID2genename", "chemicalmeshID2genename", "gwastraits2genename", "phrase2genename"]
    dictionary2genenames = [name for name in dictionaries.keys() if
                            not re.search("id$", name, re.IGNORECASE) and name not in ["priority", "stopwords",
                                                                                       "variant2genename", "readme"]]

    if priority is None:
        if "priority" in dictionaries:
            priority = dictionaries["priority"].loc[:,'priority'].to_list() + [p for p in dictionary2genenames if p not in dictionaries['priority'].loc[:, "priority"].to_list()]
            print("Dictionaries priority by-default:")
        else:
            priority = default + [p for p in dictionary2genenames if p not in default]
            print("Dictionaries priority is set as:")
    else:
        if isinstance(priority, str):
            priority = [priority]
        priority = priority + [p for p in default if p not in priority]
        print("Dictionaries priority is customized by user:")

    print("\n".join([name for name in priority if name in dictionaries]))
    dictionaries['priority'] = priority

    if not w2v_path:
        print('No word2vec file is provided. Only dictionary will be loaded.')
    else:
        print('\nReading word2vec file...')
        dictionaries['model'] = read_w2v(w2v_path)

    print(f"Total object size : {sum(os.path.getsize(p) for p in dictionary_paths) // 1024 // 1024} MB")
    if 'readme' in dictionaries and 'species' in dictionaries['readme']:
        print(f"Species : {dictionaries['readme']['species'].iloc[0, 0]}")

    return dictionaries


def tokenize(index: SCFind,
             query: str,
             strict: bool = False,
             any_id: Optional[List[str]] = None
             ) -> dict:
    if any_id is None:
        any_id = ["MESH", "CHEBI", "OMIM"]
    mode = ['AND#', 'AND_gene', 'AND_word', 'AND_variant'] if strict else ['OR#', 'OR_gene', 'OR_word', 'OR_variant']

    queries = re.split(",|;|\\. ", query)
    queries = [q for q in queries if q != ""]
    queries = ["#" + re.sub("\\s", "#", q.strip()) for q in queries]

    # Replace various phrases with AND, OR, NOT operators
    and_phrases = re.compile(
        r'#yes#|#but#also#|#also#|#along#with#|#as#well#as#|#including#|#includes#|#include#|#plus#|#with#|#keep'
        r'#|#keeping#|add#|#adding#|#both#|#contribute#to#|#connect#with#',
        re.IGNORECASE)
    or_phrases = re.compile(r'#either#|#link#to#|#associate#with#|#relate#to#', re.IGNORECASE)
    not_phrases = re.compile(
        r'#no#|#not#either#|#neither#|#nor#|#exclude#|#excluding#|#reject#|#rejecting#|#omit#|#omitting#|#lack'
        r'#|#lacking#|#minus#|#without#|#ignore#|#ignoring#|#leave#out#|#cancel#|#discard#|#drop#|#take#out#|#except'
        r'#|#excepting#|#but#not#',
        re.IGNORECASE)
    or_not_phrases = re.compile(r'#or#not#', re.IGNORECASE)

    queries = [and_phrases.sub('#AND#', q) for q in queries]
    queries = [or_phrases.sub('#OR#', q) for q in queries]
    queries = [not_phrases.sub('#NOT#', q) for q in queries]
    queries = [or_not_phrases.sub('#ORNOT#', q) for q in queries]

    queries = " # ".join([q.replace("#", " ") for q in queries])
    queries = queries.split(" ")

    # Adjustments for all_ids
    all_ids = [id + ":" for id in any_id]

    tokens = defaultdict()

    # If only gene query without operators
    if len([x for x in index._case_correct([q for q in queries if "#" not in q], if_print=False)]) == len(queries):
        tokens[mode[1]] = index._case_correct(queries, if_print=False)
        return tokens

    # Process the operators
    conds = ["AND#", "OR#", "NOT#", "ORNOT#"]
    queries_upper = [q.upper() for q in queries]
    conds_ = [cond.replace("#", "") for cond in conds]
    ops_ = [q.replace("#.*", "") for q in queries_upper if q.replace("#.*", "") in conds_]
    ops = []
    for val in ops_:
        ops.append(conds[conds_.index(val)])

    ops_inds = [i for i, q in enumerate(queries_upper) if q.replace("#.*", "") in ops_]

    if not ops:
        # Process complex query without operators
        genes = index._case_correct(queries, if_print=False)
        snps = [query for query in queries if re.match(r'^rs\d+$', query, re.IGNORECASE)]
        words = [re.sub(r'["\'`]', "", word) for word in [q.lower() for q in queries] if word not in [q.lower() for q in genes + snps]]
        tokens[mode[1]] = genes if genes else None
        tokens[mode[3]] = snps if snps else None
        seen = set()
        tokens[mode[2]] = [x for x in words if not (x in seen or seen.add(x))] if words else None
        tokens = {key: value for key, value in tokens.items() if value}
    else:
        # Process complex query with operators
        for i in range(len(ops), 0 if any(op in mode[0] for op in ops) else -1, -1):
            tk = None
            operator = ops[i - 1] if i > 0 else mode[0]

            if i == 1 and any(op in mode[0] for op in ops):
                # Take both side of operators when AND/OR are used
                tk = queries + ["#"]
                genes = index._case_correct([re.sub(ops[0], "", q, flags=re.IGNORECASE) for q in tk if not (len(q) == 1 and q in ascii_letters)], if_print=False)
                genes_lower = [gene.lower() for gene in genes]
                tk_processed = [re.sub(ops[0], "", tk_item, flags=re.IGNORECASE).lower() for tk_item in tk]
                tk_diff_genes = [item for item in tk_processed if item not in genes_lower]
                snps = [snp for snp in tk_diff_genes if re.match(r"^rs\d+$", snp)]

            else:
                if queries:
                    tk = queries[ops_inds[i - 1]:] + ["#"] if i != 0 else queries + ["#"]

                genes = index._case_correct(
                    [re.sub(operator, "", q, flags=re.IGNORECASE) for q in tk if not (len(q) == 1 and q in ascii_letters)],
                    if_print=False
                ) if tk else []
                genes_lower = [gene.lower() for gene in genes]

                tk_processed = [re.sub(operator, "", item, flags=re.IGNORECASE).lower() for item in tk]
                tk_diff_genes = [item for item in tk_processed if item not in genes_lower]
                snps = [snp for snp in tk_diff_genes if re.match(r"^rs\d+$", snp)] if tk else None

            if tk:
                operator_processed = re.sub("#", "", operator)
                del_ = [val.lower() for val in [operator_processed] + genes + snps + [""]]
                tk_processed = ([item for item in tk if item.lower() not in del_])
                words = [re.sub(r"[\"'`]", "", item) for item in tk_processed]
            else:
                words = None

            if i != 0:
                queries = queries[: ops_inds[i - 1]]

            if i == 0 and queries:
                tokens[mode[1]] = tokens.get(mode[1], []) + genes if genes else tokens.get(mode[1], [])
                tokens[mode[3]] = tokens.get(mode[3], []) + snps if snps else tokens.get(mode[3], [])
                tokens[mode[2]] = tokens.get(mode[2], []) + words if words else tokens.get(mode[2], [])
            else:
                if genes is not None or snps is not None or words is not None:
                    operator_gene = operator.replace("#", "_gene")
                    operator_variant = operator.replace("#", "_variant")
                    operator_word = operator.replace("#", "_word")

                    tokens[operator_gene] = tokens.get(operator_gene, []) + (genes if genes else [])
                    tokens[operator_variant] = tokens.get(operator_variant, []) + (snps if snps else [])
                    tokens[operator_word] = tokens.get(operator_word, []) + (words if words else [])

    if any(re.search("|".join(all_ids), token, re.IGNORECASE) for token in [t for val in tokens.values() for t in val]):
        tokens_df = DataFrame(tokens.items(), columns=['key', 'value'])
        tokens_df = tokens_df.explode('value')
        ids_inds = tokens_df['value'].str.contains("|".join(all_ids), case=False, regex=True)

        tokens_df.loc[ids_inds, 'key'] = tokens_df.loc[ids_inds, 'key'].str.replace("_word", "_mesh", regex=False)
        tokens = {key: grp['value'].tolist() for key, grp in tokens_df.groupby('key')}

    if not isinstance(tokens, dict):
        # if isinstance(tokens['res'], str):
        #     split_tokens = tokens['res'].split(',')
        if isinstance(tokens['res'], list):
            split_tokens = [item.split(',') for item in tokens['res']]
        else:
            raise ValueError("Check data type of tokens['res']")

        return {key: split_token for key, split_token in zip(tokens.keys(), split_tokens)}
    else:
        return {key: value for key, value in tokens.items() if value}


def dedup_gene(gene_list: dict) -> dict:
    genes = gene_list.get('and', []) + gene_list.get('not', []) + gene_list.get('or', []) + gene_list.get('ornot', [])

    # Find duplicate genes
    dup_genes = set([gene for gene in genes if genes.count(gene) > 1])

    if len(dup_genes) == 0:
        return gene_list
    else:
        for i in dup_genes:
            # Find all occurrences of gene_list['genes']
            occurrences = [index for index, gene in enumerate(gene_list['genes']) if re.search(i, gene, re.IGNORECASE)]

            if len(occurrences) > 1:
                kept_gene = gene_list['genes'][occurrences[0]]
                print(f"Keeping {kept_gene} in {' '.join([gene_list['genes'][idx] for idx in occurrences])}.")

                # delete duplicates in "genes"
                for idx in sorted(occurrences[1:], reverse=True):
                    del gene_list['genes'][idx]

        return gene_list


def weigh_gene(page: DataFrame,
               inds: List[int],
               greedy: Optional[float] = None
               ) -> List[str]:
    # Get gene name and frequency
    genes_ = [page.iloc[ind, 1] for ind in inds]
    genes = [g for gs in genes_ for g in gs.split(",")]
    freqs_ = [page.iloc[ind, 2] for ind in inds]
    freqs = [float(i) for nums in freqs_ for i in nums.split(",")]


    if greedy == 1:
        return genes

    cutoff = len(set(freqs)) * greedy if greedy != 0 else 1

    # Set returned genes
    unique_freqs = set(freqs)
    gene_list = [gene for gene, freq in zip(genes, freqs) if freq in sorted(set(freqs), reverse=True)[:math.ceil(cutoff)]]

    if len(gene_list) < len(genes):
        print(f"Returned {len(gene_list)} out of {len(genes)} genes (argument greedy = {greedy})")

    return gene_list


def token2phrases(dictionary: dict,
                  token: str,
                  pattern: List[str],
                  spell_tolerate: bool = False
                  ) -> DataFrame:
    inds = []
    candidates = []

    # Sort and combine token
    sorted_token = " ".join(sorted(re.split("[ ,\\-]", token)))
    all_token = f"{token}|{sorted_token}"

    for i in pattern:
        page = dictionary.get(i, None)
        if page is not None:
            if not spell_tolerate:
                if isinstance(page, dict):
                    first_key = next(iter(page))
                    tmp = [index for index, value in enumerate(page[first_key].iloc[:, 0])
                           if re.search(all_token, value, re.IGNORECASE)]
                elif isinstance(page, DataFrame):
                    tmp = [index for index, value in enumerate(page.iloc[:, 0])
                           if re.search(all_token, value, re.IGNORECASE)]
                else:
                    raise ValueError(f"Check data type of {i} in dictionary!")
            else:
                if isinstance(page, dict):
                    first_key = next(iter(page))
                    data = page[first_key].iloc[:, 0]
                elif isinstance(page, DataFrame):
                    data = page.iloc[:, 0]
                else:
                    raise ValueError("Check data type of {i} in dictionary!")

                token_lower = token.lower()
                data_lower = [str(item).lower() for item in data]

                matches = process.extract(token_lower, data_lower, scorer=fuzz.WRatio, score_cutoff=80, limit=None)
                tmp = [index for _, _, index in matches]

            if tmp:
                inds.extend([f"{i}_{t}" for t in tmp])
            if isinstance(page, dict):
                first_key = next(iter(page))
                candidates.extend(page[first_key].iloc[tmp, 0])
            else:
                candidates.extend(page.iloc[tmp, 0])

    if inds:
        df = DataFrame({
            "dictionary": [re.sub("\\$_.*|_.*", "", ind) for ind in inds],
            "id": [int(re.sub(".*_", "", ind)) for ind in inds],
            "phrase": candidates
        })
    else:
        df = DataFrame()

    if any(df['phrase'].isin([token, sorted_token])):
        df = df[df['phrase'].isin([token, sorted_token])]
        df.loc[:,'phrase'] = [",".join(sorted(phrase.strip().split(','))) for phrase in df['phrase']]

        if df['phrase'].duplicated().any():
            df = df.iloc[[df['dictionary'].apply(lambda x: pattern.index(x)).argmin()]]

        print(f"Found '{token}' in {df.loc[df['phrase']==token, 'dictionary'].values[0]} dictionary.")

    return df


def id2genes(dictionary: dict,
             bestmatch: DataFrame,
             greedy: float
             ) -> Union[None, List[str]]:
    if bestmatch.shape[1] == 0:
        return None

    if bestmatch.shape[0] > 1:
        print("Warning: argument 'bestmatch' has number of rows > 1 and only the first row will be used\n")
        bestmatch = bestmatch.iloc[0, :]

    page = bestmatch['dictionary'][0]

    if isinstance(dictionary[page], dict):
        first_key = next(iter(dictionary[page]))
        mesh_id = dictionary[page][first_key].iloc[bestmatch['id'][0], 1].split()[0]
        if ":" not in mesh_id:
            mesh_id = ":" + mesh_id

        # replace by row number in meshID2genename dictionary
        mID = list(dictionary[page].values())[1]
        bestmatch_id = [i for i, item in enumerate(mID.iloc[:, 0]) if mesh_id in item]
        if bestmatch_id is None:
            print(f"No relevant gene for '{bestmatch['phrase']}'")
            return None
        else:
            return weigh_gene(page=mID, inds=bestmatch_id, greedy=greedy)


def cos_sim_Word2Vec(model: KeyedVectors,
                     tokens: List[str],
                     candidates: List[str]
                     ) -> dict:
    similarities = {}

    for token in tokens:
        dists = []
        for candidate in candidates:
            try:
                dist = model.similarity(token, candidate)
            except KeyError:
                dist = -1
            dists.append(dist)
        similarities[token] = dists

    similarities['similarity'] = np.sum(list(similarities.values()), axis=0)
    similarities['candidates'] = candidates

    return similarities


def do_u_mean(candidates: List[str],
              view_all=False
              ):
    if not view_all:
        pos = ["", "y", "yes", "ok", "pos", "positive"]
        neg = ["n", "no", "neg", "negative"]
        cancel = ["c", "cancel", "stop", "break"]

        for i, candidate in enumerate(candidates, start=1):
            result = input(f"Do you mean '{candidate}'? (y/n/c): ").lower()
            if any(word in result for word in pos):
                print(f"User chose {candidate}")
                return candidate
            if any(word in result for word in cancel):
                print("Warning: This word will be neglected.")
                return None
            if any(word in result for word in neg):
                continue

        print("No match found, this word will be neglected.")
        return None

    else:
        print("Do You Mean?")
        for i, candidate in enumerate(candidates, start=1):
            print(f"[{i}] {candidate}")

        try:
            result = int(input("Please input a number: "))
            if 0 < result <= len(candidates):
                return candidates[result - 1]
        except ValueError:
            pass

        print("Invalid input or no match found.")
        return None


# Example usage
# candidates = ['word1', 'word2', 'word3']
# choice = do_u_mean(candidates, view_all=False)


def query2genes(index: SCFind,
                dictionary: dict,
                query: str,
                strict: bool = False,
                automatch: bool = True,
                greedy: float = 0.6,
                priority: Optional[List[str]] = None,
                spell_tolerate: bool = True
                ) -> dict:
    if 'model' in dictionary:
        model = dictionary['model']
    else:
        raise ValueError('No word2vec model is provided in your dictionaries.')

    query = tokenize(index, query, strict)

    gene_list = []
    raw_genes = []
    genes = None
    phrase = None
    weight = None
    conds = ["AND_", "OR_", "NOT_", "ORNOT_"]
    symbl = ["", "*", "-", "*-"]

    # Initialize results dictionary
    results = {}
    # If only gene query without operators
    if "AND_gene" in query and len(query) == 1:
        results["genes"] = query["AND_gene"]
        return results

    if any(re.search("_gene", key) for key in query):
        # Get gene_list
        gene_list = []
        gene_list.extend(query["AND_gene"] if "AND_gene" in query else [])
        gene_list.extend(["*" + or_gene for or_gene in query['OR_gene']] if "OR_gene" in query else [])
        gene_list.extend(["-" + not_gene for not_gene in query['NOT_gene']] if "NOT_gene" in query else [])
        gene_list.extend(["*-" + ornot_gene for ornot_gene in query['ORNOT_gene']] if "ORNOT_gene" in query else [])

        # Get results
        results = {
            "and": query["AND_gene"] if "AND_gene" in query else [],
            "or": query["OR_gene"] if "OR_gene" in query else [],
            "not": query["NOT_gene"] if "NOT_gene" in query else [],
            "ornot": query["ORNOT_gene"] if "ORNOT_gene" in query else []
        }

        # If all gene query
        if all("_gene" in key for key in query):
            results["genes"] = gene_list
            results = dedup_gene(results)
            # Remove empty elements
            return {k: v for k, v in results.items() if v}

    if any(re.search("variant", key, re.IGNORECASE) for key in dictionary) and any(
            re.search("variant", key, re.IGNORECASE) for key in query):
        # Get the variant page in dictionary
        pages = [key for key in dictionary if "variant" in key.lower()]

        for i in [key for key in query if "_variant" in key.lower()]:
            raw_genes = index._case_correct(weigh_gene(page=dictionary[pages[0]],
                                                       inds=[index for index, value in enumerate(dictionary[pages[0]].iloc[:, 0])
                                                             if value in query[i]], greedy=greedy), if_print=False)
            results_key = i.replace("_variant", "").lower()
            results[results_key] = results.get(results_key, []) + raw_genes

            if 'AND_variant' not in i and len(raw_genes) != 0:
                pattern = re.sub("variant", "", i)
                matched_indices = [index for index, value in enumerate(conds) if re.search(pattern, value)]
                symbol = symbl[matched_indices[0]]
                raw_genes = [symbol + gene for gene in raw_genes]

            gene_list += raw_genes if raw_genes else []

        if not any("_word" in key.lower() or "_mesh" in key.lower() for key in query):
            seen = set()
            results["genes"] = [x for x in gene_list if not (x in seen or seen.add(x))]  # remove duplicates
            results = dedup_gene(results)
            return {k: v for k, v in results.items() if v}

    if any(re.search("meshID2genename", key, re.IGNORECASE) for key in dictionary) and any(
            re.search("mesh", key, re.IGNORECASE) for key in query):
        pages = [key for key in dictionary if re.search("meshID2genename", key, re.IGNORECASE)]
        for page in pages:
            for i in [key for key in query if re.search("_mesh", key, re.IGNORECASE)]:

                raw_genes = index._case_correct(weigh_gene(page=dictionary[page]['2'],
                                                           inds=[index for index, value in
                                                                 enumerate(dictionary[page]['2'].iloc[:, 0])
                                                                 if value in [q.upper() for q in query[i]]],
                                                           greedy=greedy),
                                                if_print=False)

                results_key = re.sub("_mesh", "", i).lower()
                results[results_key] = results.get(results_key, []) + raw_genes

                if not re.search('AND_mesh', i) and len(raw_genes) != 0:
                    pattern = re.sub("mesh", "", i)
                    matched_indices = [index for index, value in enumerate(conds) if re.search(pattern, value)]
                    symbol = symbl[matched_indices[0]]
                    raw_genes = [symbol + gene for gene in raw_genes]

                gene_list += raw_genes if len(raw_genes) != 0 else []

        if not any(re.search('_word', key) for key in query):
            seen = set()
            results["genes"] = [x for x in gene_list if not (x in seen or seen.add(x))]  # remove duplicates
            results = dedup_gene(results)
            return {k: v for k, v in results.items() if v is not None}

    df = DataFrame([(k, v) for k, vals in query.items() for v in vals], columns=['ind', 'values'])
    df = df[df['ind'].str.contains('_word')]
    df = df[df['values'] != ""]

    # Clean stop words
    stop_keys = [key for key in dictionary if re.search("stop", key, re.IGNORECASE)]
    if stop_keys:
        stop_words = [word for key in stop_keys for word in dictionary[key]]
        stop_inds = [index for index, value in enumerate(df['ind']) if value in stop_words]
        if stop_inds:
            df = df.drop(stop_inds)
    if priority:
        pattern = priority + [item for item in dictionary.get("priority", []) if item not in priority]
    else:
        pattern = [item for item in dictionary.get('priority', [])]

    if not df.empty and any(re.search("|".join(pattern), key, re.IGNORECASE) for key in dictionary):
        bestmatch = DataFrame(columns=['dictionary', 'id', 'phrase'])

        unique_words = df[df.loc[:, 'ind'].str.contains("_word")].loc[:, 'ind'].unique()
        for i in unique_words:
            tokens = df[df['ind'] == i].loc[:, 'values'].tolist()
            tokens = re.split(r' # |#| #|# ', ' '.join(tokens))
            # Remove empty space
            tokens = [token for token in tokens if token]

            if tokens:
                for j in tokens:
                    res = token2phrases(dictionary=dictionary, token=j, pattern=pattern, spell_tolerate=spell_tolerate)
                    if len(res) == 1:
                        raw_genes.extend(id2genes(dictionary=dictionary, bestmatch=res, greedy=greedy))
                        print(f"Found {', '.join(raw_genes)} for {j}")
                    else:
                        if len(res) == 0:
                            tmp_tk = j.split(" ")
                            if len(tmp_tk) == 1:
                                raw_genes = None
                                print(f"No gene found for '{j}'")
                            else:
                                cands = token2phrases(dictionary=dictionary, token=re.sub(r"\s", "|", j),
                                                      pattern=pattern)
                                cands['phrase'] = [' '.join(sorted(phrase.split()))
                                                   for phrase in cands['phrase'].astype(str)]

                                for k in sorted(tmp_tk):
                                    cands_tk = cands[cands['phrase'].str.contains(k, case=False, na=False)][
                                        'phrase'].tolist()
                                    if len(cands_tk) == 0:
                                        print(f"No gene found for '{k}'")
                                    else:
                                        bestmatch_phrases = [' '.join(sorted(phrase.split(" "))) for phrase in
                                                             bestmatch['phrase']]
                                        if not any(phrase in cands_tk for phrase in bestmatch_phrases):
                                            cands_tk = DataFrame(
                                                [(phrase, token) for phrase in cands_tk for token in
                                                 re.split(r'[ \-,]', phrase)],
                                                columns=['phrase', 'token'])
                                            unique_tokens = cands_tk['token'].unique()
                                            cands_sim = cos_sim_Word2Vec(model, j.split(" "), unique_tokens)
                                            cands_tk['similarity'] = cands_tk['token'].map(
                                                lambda x: cands_sim['similarity'][
                                                    list(cands_sim['candidates']).index(x) if x in cands_sim[
                                                        'candidates'] else None])
                                            cands_tk = cands_tk.groupby('phrase').similarity.mean().reset_index()
                                            cands_tk = cands_tk.sort_values(by='similarity', ascending=False)

                                            best_match_phrase = cands_tk.iloc[0][
                                                'phrase'] if not cands_tk.empty else None
                                            if best_match_phrase:
                                                bestmatch = concat(
                                                    [bestmatch, cands[cands['phrase'] == best_match_phrase]])
                                print(f"Found '{', '.join(bestmatch['phrase'])}' for '{j}'.")
                        else:
                            if isinstance(j, str) and spell_tolerate:
                                words = re.split(r'[ \-,]', ' '.join(res['phrase']))
                                tk_in_common = Counter(words).most_common(1)[0][0]
                                if re.search(j, tk_in_common, re.IGNORECASE) and Counter(words)[tk_in_common] == len(
                                        res):
                                    j = tk_in_common

                            cands_tk = {phrase: re.split(r'[ \-,]', phrase) for phrase in res['phrase']}

                            if len(cands_tk) > 1:
                                cands_tk = DataFrame(
                                    [(key, val) for key, values in cands_tk.items() for val in values],
                                    columns=['phrase', 'token'])
                                unique_tokens = cands_tk['token'].unique()
                                cands_sim = cos_sim_Word2Vec(model, j.split(" "), unique_tokens)

                                similarity_map = dict(zip(cands_sim['candidates'], cands_sim['similarity']))
                                cands_tk['similarity'] = cands_tk['token'].map(similarity_map.get)

                                cands_tk = cands_tk.groupby('phrase')['similarity'].mean().reset_index()
                                cands_tk = cands_tk.sort_values(by='similarity', ascending=False)

                                if automatch:
                                    bestmatch = res[res['phrase'] == cands_tk.iloc[0]['phrase']].iloc[0]
                                    print(f"Found '{bestmatch['phrase']}' for '{j}'.")
                                else:
                                    # bestmatch < - res[which(res$phrase % in %do.u.mean(cands.tk, F))[1],]
                                    bestmatch = res[res[
                                        'phrase' == do_u_mean(cands_tk, False)]]  # adjust based on result of du_u_mean
                            else:
                                bestmatch = res

                        if bestmatch.ndim == 1:
                            bestmatch = pd.DataFrame([bestmatch]).reset_index(drop=True)
                        for k in range(len(bestmatch)):
                            bestmatch_k = pd.DataFrame([bestmatch.iloc[k]]).reset_index(drop=True)
                            raw_genes.extend(id2genes(dictionary=dictionary,
                                                      bestmatch=bestmatch_k, greedy=greedy))

                genes = index._case_correct(raw_genes, if_print=False)
                key = i.replace("_word", "").lower()
                results[key] = results.get(key, []) + genes

                if len(genes) > 0:
                    symbol = symbl[conds.index(i.replace("word", ""))] if i.replace("word", "") in conds else ""
                    genes = [f"{symbol}{gene}" for gene in genes]

                if genes:
                    gene_list.extend(genes)

        seen = set()
        results['genes'] = [x for x in gene_list if not (x in seen or seen.add(x))]
        results = dedup_gene(results)
        return {key: val for key, val in results.items() if val}

    else:
        results['genes'] = gene_list if len(gene_list) != 0 else None
        results = dedup_gene(results)
        return {key: val for key, val in results.items() if val is not None}


def query2CellTypes(index: SCFind,
                    dictionary: dict,
                    query: str,
                    datasets: Optional[Union[str, List[str]]] = None,
                    optimize=True,
                    abstract=True,
                    strict=False,
                    greedy=0.6,
                    priority=None,
                    spell_tolerate=True) -> dict:
    if datasets is None:
        datasets = index.datasets
    else:
        datasets = index._select_datasets(datasets)

    optimized_query = None
    result = {}
    gene_list = query2genes(index=index, dictionary=dictionary, query=query, strict=strict, greedy=greedy,
                            priority=priority, spell_tolerate=spell_tolerate)

    if optimize and any("and" in name or "or" in name for name in gene_list):

        key_and_or = [key for key in gene_list if re.match("^and$|^or$", key)]
        for i in key_and_or:
            if len(gene_list[i]) > 1:
                optimized_query = index.markerGenes(gene_list[i])
                optimized_query = optimized_query.sort_values(by='tfidf', ascending=False).iloc[0]['Query']
                optimized_query = optimized_query.split(',')
            else:
                optimized_query = gene_list[i]

            s = ["*" + item if i == 'or' else item for item in [g for g in gene_list[i] if g not in optimized_query]]
            gene_list['genes'] = [g for g in gene_list['genes'] if g not in s]
            gene_list[i] = optimized_query if optimized_query else gene_list[i]

        result['query_optimized'] = gene_list
    else:
        result['query'] = gene_list

    cell_hits = index.findCellTypes(gene_list['genes'], datasets)

    if cell_hits:
        q2CT = index._phyper_test(cell_hits)
    else:
        print("No Cell Is Found!")
        return result

    print(f"Found {len(q2CT)} cell types for your search '{re.sub(',.*', '', query)}...'.")

    if not abstract:
        q2CT = q2CT.sort_values('pval')
        result['cell_hits'] = cell_hits
        result['celltypes'] = q2CT
        return result
    else:
        significant_celltype = q2CT[q2CT['pval'] <= min(q2CT['pval'].min(), 0.001)]
        if not significant_celltype.empty:
            result['celltypes_significant'] = significant_celltype
            cell_hits = {key: cell_hits[key] for key in significant_celltype['cell_type']}
            result['cell_hits_significant'] = cell_hits
            print(f"Found {len(significant_celltype)} cell types has pval <= {min(q2CT['pval'].min(), 0.001)}.")
        else:
            print("No Significant CellType Is Found!")

    return result