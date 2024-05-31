# Hugging Face Transformer
This project implements a Transformer encoder from scratch to explore how they work.
Then it uses an encoder-decoder model for semantic parsing, using Huggingface Transformers, a popular open-source library for NLP.
Different inference methods are then explored for this model.

# Task 1:
1.) Implement a simplified Transformer (missing components like layer normalization and multi-head attention)
from scratch for a simple task. Given a string of characters, the task is to predict, for each position in the string,
how many times the character at that position occurred before, maxing out at 2.

2.) Extend your Transformer classifier with positional encodings and address the main task: identifyingthenumberoflettersofthesametypeprecedingthatletter.Runthiswithpython lettercounting.py, no other arguments. Without positional encodings, the model simply sees a bag of characters and cannot
distinguish letters occurring later or earlier in the sentence (although loss will still decrease and something can still be learned).
Provided: a PositionalEncoding module that you can use: this initializes a nn.Embedding layer, embeds the index of each character, then adds these to the actual character embeddings.1 If the input sequence is the, then the embedding of the first token would be embedchar(t) + embedpos(0), and the embedding of the second token would be embedchar(h) + embedpos(1).
The final implementation should get over 95% accuracy on this task.

# Task 2: Semantic Parsing with Seq2seq Models

Semantic parsing involves translating sentences into various kinds of formal representations such as lambda calculus or lambda-DCS. These representations’ main feature is that they fully disambiguate the natural language and can effectively be treated like source code: executed to compute a result in the context of an environment such as a knowledge base. In this case, you will be dealing with the Geoquery dataset (Zelle and Mooney, 1996). Two examples from this dataset formatted as you’ll be using are shown below:
what is the population of atlanta ga ?
_answer ( A , ( _population ( B , A ) , _const ( B , _cityid ( atlanta , _ ) ) ) )
what states border texas ?
_answer ( A , ( _state ( A ) , _next_to ( A , B ) , _const ( B , _stateid ( texas ) ) ) )
These are Prolog formulas similar to lambda calculus expressions. In each case, an answer is computed by executing this expression against the knowledge base and finding the entity A for which the expression evaluates to true.
This task will follow in the vein of Jia and Liang (2016), who tackle this problem with sequence-to- sequence models. These models are not guaranteed to produce valid logical forms, but circumvent the need to come up with an explicit grammar, lexicon, and parsing model. In practice, encoder-decoder models can learn simple structural constraints such as parenthesis balancing (when appropriately trained), and typically make errors that reflect a misunderstanding of the underlying sentence, i.e., producing a valid but incorrect logical form, or “hallucinating” things that weren’t there.
We can evaluate these models in a few ways: based on the denotation (the answer that the logical form gives when executed against the knowledge base), based on simple token-level comparison against the ref- erence logical form, and by exact match against the reference logical form (slightly more stringent than denotation match).
The data consists of a sequence of (example, logical form) sentence pairs. geo train.tsv con- tains a training set of 480 pairs, geo dev.tsv contains a dev set of 120 pairs, and geo test.tsv
contains a blind test set of 280 pairs (the standard test set). This file has been filled with junk logical forms (a single one replicated over each line) so it can be read and handled in the same format as the others.
There are a few more steps taken to convert to a usable format for HuggingFace training. First, convert to hf datase pads the inputs and labels to be square tensors, one input example per row in the input tensor and one target
decoder output per row in the labels. The longest input is 23, the longest output is 65. With a BART model, -100 is the output pad index.
Second, we prepare an attention mask. This is a matrix the same size as the
inputs with 1s in positions corresponding to “real” inputs and 0s in positions corresponding to pad inputs (e.g., those past the length of the corresponding input example). Note that because of this matrix, the input pad index shouldn’t actually matter because attention to these tokens is zeroed out and they will not impact the decoder.
We then returns a dict consisting of three things:
encodings = {’input_ids’: inputs, ’attention_mask’: attention_mask, ’labels’: labels}
BART: We base our code on the BartForConditionalGeneration model from HuggingFace.
However, we are just using this configuration for simplicity; we are not actually using the BART pre-trained
weights. Setting aside the pre-training, BartForConditionalGeneration is an encoder-decoder Transformer model in the style of (Vaswani et al., 2017). Our hyperparameters give a
modest-sized model that is nevertheless sufficient to fit this dataset.

# Task 3: decode oracle and decode fancy

This tasks explores possible extensions to basic decoding of seq2seq models with two extensions.
Both of these approaches will center around reranking. Reranking involves getting a set of n options (usually from beam search) and returning one that may not have scored the highest under the original model
using some sort of auxiliary objective.
Oracle: The first task in reranking is to compute an oracle over the beam. An oracle is essentially a “cheating” reranker: you should look at the options returned from beam search, score them, and return the one with the highest score. This is an effective debugging tool: it tells you whether your general reranking code is working correctly and can improve accuracy. Furthermore, it tells you what the theoretical limit of a reranker is: if you know that the oracle performance is only 60%, then there’s no way that any reranker you implement could achieve better than that.
The general steps are to generate with beam search, iterate through the beam, score each of the options against the gold standard (this is the “cheating” part), and return the option with the best score.
Fancy:  Second, I implement a real, non-oracle reranker over the beam. I focus on rule-based constraint: making sure the right expressions or literal constants are used. Many errors are of the form “what states border texas” and then the generated logical form contains the word “california”. By preferring options in the beam that use the correct literal constants (texas in this case)
The final “fancy” gets an exact match of above 51% on the development set, and the minimum goal for oracle approach should get above 55%.
