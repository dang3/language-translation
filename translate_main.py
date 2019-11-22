import translate_utils as utils

# sentences used to train
eng = ""
fra = ""

eng_word2int = ""
en_word2int_mapper = ""
fra_word2int = ""
fra_word2int_mapper = ""


eng, fra = utils.create_dataset('eng.txt', 'fra.txt', 10)
eng_word2int, eng_word2int_mapper = utils.tokenize(eng)
fra_word2int, fra_word2int_mapper = utils.tokenize(fra)

batch_size = 64

