def merge_dict_and_g2p_outputs(dict_path, g2p_outputs_path, merged_result_path):
    existing_words = open(dict_path).readlines()
    with open(g2p_outputs_path, encoding="us-ascii") as f2, open(merged_result_path, 'w', encoding="us-ascii") as t:
        samples = f2.readlines()
        print(len(samples))
        for i, sample in enumerate(samples):
            tmp = sample.split("\t")
            word = tmp[0]
            transcription = tmp[1]
            existing_words.append('\t'.join((word, str(1.0), str(0.0), str(0.0), str(0.0), transcription)))
        t.write(''.join(existing_words))


if __name__ == "__main__":
    dict_path = "/root/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict"
    g2p_outputs_path = "/root/Documents/MFA/pretrained_models/dictionary/english_emo_dictionary.txt"
    merged_result_path = "/root/Documents/MFA/pretrained_models/dictionary/english_emodb_parallel.txt"
    merge_dict_and_g2p_outputs(dict_path, g2p_outputs_path, merged_result_path)
