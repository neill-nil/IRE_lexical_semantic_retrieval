#!/bin/bash
echo "Downloading Ekstra_Bladet_word2vec.zip..."
wget -c https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip

echo "Downloading google_bert_base_multilingual_cased.zip..."
wget -c https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip

echo "Downloading FacebookAI_xlm_roberta_base.zip..."
wget -c https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/FacebookAI_xlm_roberta_base.zip

echo "Downloads complete."
