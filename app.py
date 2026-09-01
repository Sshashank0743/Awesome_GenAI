import sacrebleu

# 1. Define your model outputs and the ground-truth references
system_outputs = ["The cat is on the mat.", "There is an airplane in the sky."]
references = [
    ["The cat is sitting on the mat.", "An airplane is flying in the sky."]
]

# 2. Compute the corpus-level BLEU score
bleu = sacrebleu.corpus_bleu(system_outputs, references)

# 3. Output the exact score and the official shareable signature
print(f"BLEU Score: {bleu.score:.2f}")

# FIX: Use bleu.format(signature=True) to extract the signature format string
print(f"Official SacreBLEU Signature: {bleu.format(signature=True)}")
