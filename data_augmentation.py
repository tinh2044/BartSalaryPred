import random
import numpy as np

def random_deletion(sentence, prob=0.1):
    """Randomly delete words from the sentence with probability prob."""
    words = sentence.split()
    if len(words) == 1:
        return sentence
    return ' '.join([word for word in words if random.random() > prob])
    
def random_insertion(sentence, prob=0.1):
    """Randomly insert words from the sentence into the sentence with probability prob."""
    words = sentence.split()
    n = len(words)
    for _ in range(int(prob * n)):
        idx = random.randint(0, n-1)
        insert_idx = random.randint(0, n)
        words.insert(insert_idx, words[idx])
    return ' '.join(words)
    
def add_noise(sentence, prob=0.05):
    """Add character-level noise to the sentence by swapping adjacent characters."""
    def swap_chars(word):
        if len(word) > 1:
            idx = random.randint(0, len(word) - 2)
            return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
        return word
    
    words = sentence.split()
    noisy_words = [swap_chars(word) if random.random() < prob else word for word in words]
    return ' '.join(noisy_words)

def augment_data(text):
    """Apply a randomly selected augmentation to the input text."""
    idx = np.random.randint(1, 4)
    if idx == 1:
        return random_deletion(text, np.random.uniform(0.1, 0.5))
    elif idx == 2:
        return random_insertion(text, np.random.uniform(0.1, 0.5))
    elif idx == 3:
        return add_noise(text, np.random.uniform(0.1, 0.4)) 