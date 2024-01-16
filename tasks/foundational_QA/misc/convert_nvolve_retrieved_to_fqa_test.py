
import json


def convert(input_datapath, retrieved_datapath, output_datapath):
    print("loading from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        data_list = json.load(f)
    
    print("loading from %s" % retrieved_datapath)
    retrieved_list = []
    with open(retrieved_datapath, "r") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            retrieved_list.append(item)
    
    assert len(data_list) == len(retrieved_list)
    output_list = []
    for data_item, retrieved_item in zip(data_list, retrieved_list):
        question = data_item['question']
        assert question == retrieved_item[0]
        ctxs = retrieved_item[1]
        if "answers" in data_item:
            data_dict = {
                "ctxs": ctxs,
                "question": question,
                "answers": data_item['answers'],
                "sub-paragraphs": ""
            }
        else:
            data_dict = {
                "ctxs": ctxs,
                "question": question,
                "answer": data_item['answer'],
                "sub-paragraphs": ""
            }
        output_list.append(data_dict)
    
    print("writing to %s" % output_datapath)
    with open(output_datapath, "w") as f:
        json.dump(output_list, f, indent=2)


def main():
    # ## att
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/pengx/data/att/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    # retrieved_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages/att_retrieval.json"
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages/att_test.json"
    # convert(input_datapath, retrieved_datapath, output_datapath)

    # ## nvit
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    # retrieved_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages/nvit_retrieval.json"
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages/nvit_test.json"
    # convert(input_datapath, retrieved_datapath, output_datapath)

    # ## iternal
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    # retrieved_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages/iternal_retrieval.json"
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages/iternal_test.json"
    # convert(input_datapath, retrieved_datapath, output_datapath)

    ## sandia
    input_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/sandia_e5_unsupervised_retriever_chunkbysents300_retrieved//test.json"
    retrieved_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages/sandia_retrieval.json"
    output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages/sandia_test.json"
    convert(input_datapath, retrieved_datapath, output_datapath)



if __name__ == "__main__":
    main()

