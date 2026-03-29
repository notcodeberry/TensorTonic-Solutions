def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    dict = {}
    
    """
    for sent in sentences:
        for word in sent:
            if word in dict:
                i = dict[word]
                dict[word] = i + 1 
            else:
                dict[word] = 1 
    """
    
    for sent in sentences:
        for word in sent:
            dict[word] = dict.get(word, 0) + 1
    
    return dict