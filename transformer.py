import requests
import numpy as np
from maths_attention import multihead_attention, positional_encoding
from transformer_layer import transformer
import re


#function to tokenize chunks and training data
def tokenizer(chunk):
    words=[]
    chunk=chunk.lower()
    for word in chunk:
        if word.isalnum() or word.isspace():
            words.append(word)
    
    words=''.join(words)
    return words.split()


#this  function will create vocab from training data: vocab = dict that assigns numbers to tokens.
#it will act as global vocabulary and also to assign number to chunks tokens
def vocab(tokens,max_vocab=None):
    uniq=[]
    for word in tokens:
        if word not in uniq:   #removing duplicates
            uniq.append(word)
    
    vocab_dict={} #it will contain words as key and assigned number(index in this case)
    for idx,word in enumerate(uniq,1): #started from 1 bcs we will reserved 0 for every unknown word in chunks from input document
        vocab_dict[word]=idx

    return vocab_dict

#this function will convert words chunks[hi,hello,world] to number matrix with number assigned to words from training data
def token_to_id(token,vocab):

    token_ids=[]

    for ch in token:
        token_ids.append(vocab.get(ch,0))

    return token_ids

#this function only cleans that that we get for wikipedia, like removing stuff that cause confusions while tokenization

def clean_wikitext(text):
    text = re.sub(r'{{.*?}}', ' ', text)   # remove {{ }} templates
    text = re.sub(r'\[\[.*?\]\]', ' ', text)  #remove [[ ]] links
    text = re.sub(r'\[.*?\]', ' ', text)  # remove [ ] links
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # remove special chars
    return text

#this will will get vocab for data used for training
def training_data():
    
    urls = [
    "https://en.wikipedia.org/wiki/Dark_energy?action=raw",
    "https://en.wikipedia.org/wiki/Cosmology?action=raw",
    "https://en.wikipedia.org/wiki/Universe?action=raw",
    "https://en.wikipedia.org/wiki/Dark_matter?action=raw"
    ]
    headers = {"User-Agent": "MyRAGProject/1.0 (student project)"}
    all_text=""
    for url in urls:
        raw_text = requests.get(url,headers=headers).text
        clean=clean_wikitext(raw_text)
        all_text+=clean

    tokens=tokenizer(all_text)  #getting tokens for data

    vocabulary=vocab(tokens)

    return vocabulary


#function to do chunks of document and will tokenize them
def chunks_to_tokens(docs,chunk_size=512): #chunk size 512 is arbitary default
    all_chunks=[]
    all_chunks_text=[]

    for chunk in range(0,len(docs),chunk_size):
        chunk_text=docs[chunk:chunk_size+chunk]
        all_chunks_text.append(chunk_text)
        chunk_token=tokenizer(chunk_text)   #making tokens of each chunks
        all_chunks.append(chunk_token)

    return all_chunks_text,all_chunks



def final_transformer_output(text,vocabulary,E,chunk_size=512):
    

    all_chunks_text,all_chunks_tokens=chunks_to_tokens(text,chunk_size)  #getting all data distributed in chunks

    all_chunks_output=[] #final output of all chunks from transformer will be collected in it 

    for chunk_tokens in all_chunks_tokens:

        token_ids=token_to_id(chunk_tokens,vocabulary)
        X=E[token_ids]
        out=transformer(X)
        
        all_chunks_output.append(out)  #we get a list now we want metrix

    final_matrix=np.stack(all_chunks_output)
    total_chunks=len(all_chunks_output)

    return all_chunks_text,final_matrix,total_chunks


#this function will embed query string
def query_str_to_vec(query,vocabulary,E):

    tokens=tokenizer(query)
    token_id=token_to_id(tokens,vocabulary)
    X=E[token_id]
    query_Vec=transformer(X)

    return query_Vec













