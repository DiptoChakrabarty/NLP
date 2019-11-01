import nltk
text = "I am going to have ice cream"
text=text.lower()
text=nltk.tokenize.word_tokenize(text)
print(text)
new_dict={}
group=[]
for i in range(len(text)-2):
    list_word=[text[i-1],text[i+1]]
    group.append(list_word)
    if text[i] not in new_dict:
        new_dict[text[i-1],text[i+1]]={}
        new_dict[text[i-1],text[i+1]][text[i]]=1
    else:
        new_dict[text[i-1],text[i+1]][text[i]]+=1
print(group)
print(new_dict)
    