def merge_dict_and_g2p_outputs(dict_path, g2p_outputs_path, merged_result_path):
    existing_words = open(dict_path).readlines()
    with open(g2p_outputs_path) as f2, open(merged_result_path, 'w') as t:
        samples = f2.readlines()
        print(len(samples))
        for i, sample in enumerate(samples):
            tmp = sample.strip().split()
            word = tmp[0]
            phonemes = ' '.join(tmp[1:]).replace(",", "").replace("'", "")[1:-1]
            existing_words.append('\t'.join((word, str(1.0), str(0.0), str(0.0), str(0.0), phonemes)) + '\n')

        t.write(''.join(existing_words))


if __name__ == "__main__":
    dict_path = "/root/Documents/MFA/pretrained_models/dictionary/russian_mfa.dict"
    g2p_outputs_path = "/root/storage/dasha/new_dictionary.txt"
    merged_result_path = "/root/Documents/MFA/pretrained_models/dictionary/sobchak_russian.txt"
    merge_dict_and_g2p_outputs(dict_path, g2p_outputs_path, merged_result_path)
