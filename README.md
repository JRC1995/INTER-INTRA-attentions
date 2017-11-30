# Intra-Layer Attention + Inter-Layer Attention + seq2seq

This is a similar seq-2-seq model as [this](https://github.com/JRC1995/Abstractive-Summarization) ,used on [Amazon-Fine-Food-Dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews/data) for abstractive summarization
(and should be usable on translation data for machine translation). 

The main difference is that the encoder and decoder units uses an intra-layer attention mechanism inspired from the idea of [recurrent residual attention](https://arxiv.org/abs/1709.03714). 

The RRA model adds a weighted summation of K previous hidden states with the standard recurrent function. 
However, unlike the standard attention where the attention weights are computed from the context by a scoring\compatibility function, in the RRA model, the weights are randomly initialized i.e they are completely parametric.

So, I modified the RRA, so that the previous hidden states are weighted based on the scores of a compatibility function or a scoring function (that scores all past hidden states in the context of the current candidate hidden state). Essentially I am using  [global attention](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf) but in-between temporally separated hidden states of the same RNN unit instead of using it between encoder and decoder layers. Which is why I am calling it 'intra-layer-attention'. 

In addition, I also used inter-layer attention (global attention). 

<b>Note: There may have been previous works on intra-layer-attention, but I didn't refer to any such work while making this implementation. This is my personal implementation, and as such it may have differences from other works in the literature that goes by that uses 'intra attention', or something similar motivated by a similar concept. </b>

# Intra-layer-attention

At each time step, a <i>candidate hidden state</i> is created by the standard RNN method.

>candidate_hidden = activate(input * wxh + previous_hidden_state * whh + Bias)

Now for this intra-layer-attention, all the previous hidden states can be considered as analogous to the encoder hidden states for the layerwise-attention (inter-layer attention - the one that is commonly used), and the candidate_hidden_state can be considered as analogous to the current decoder hidden state for the same (inter-layer attention). 

All the previous hidden states are then scored by the scoring function or the compatibility function taking the current candidate hidden state and the past hidden states as arguments. 

The scoring function, I used is: 
> hs * Transpose(ht)

ht = traget hidden, and hs = source hidden. 
In this context, ht will be the current candidate hidden state, and hs will be the list of all previous hidden states.

Similar to the standard inter-layer global attention mechanism, a context vector is created as a result of the weighted summation of all the previous hidden states, and then an 'attended hidden state' is created by taking both the context vector and the candidate hidden state into account (following the formula for creating ht_dash as given in [here](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf))

This intra-decoder-layer attended hidden state is considered as the final hidden state for that timestep. 

Finally, I stored the computed hidden state in the list of previous hidden states for computing the next hidden state in the next time step.

<b>Possible Improvement</b>: Slightly change (using some proper function) previous hidden states that were attended taking the current context into account. This is based on the intuition that while reading a text, previously read words can take on new meaning from context of the current words. Often we may look back previous words to make sense of the current word. That can be somewhat analogous to the intra-layer-attention - we attend to certain previous words in the text (or in our memory) to make sense of the current words. But, as we attend to previous words, the previous words can make more sense too, from the context of the present words. Which is why, updating previous hidden states based on current hidden states may be something to be explored (if not already explored). We already have bi-directional RNNs (which is what I used here, too) which are based on similar intuitions - that future contexts matter. But, this proposed improvement may eliminate the need for backward RNN, and may learn to update only the relevant hidden state (previous context). Also this model may be better able to handle distant context. 

However, I am unsure, if this truly will make too much of a difference.  

# Is it any better?

Can't say. I didn't have much computational resource when I developed the model. I may properly test it later. I will probably modify the model, and polish the code too. 

Ultimately, this is pretty much a toy implementation, that will work only for batch_size = 1. No regularization is implemented either. 

Example output:

>Iteration: 3
>Training input sequence length: 32
>Training target outputs sequence length: 2
>
>TEXT:
>great taffy at a great price. there was a wide assortment of yummy taffy. delivery was very quick. if your a taffy lover, >this is a deal.
>
>
>PREDICTED SUMMARY:
>
>cough cottage
>
>ACTUAL SUMMARY:
>
>great taffy
>
>loss=10.4148

Nothing to expect here. This is just the result of the 3rd iteration (i.e trained only on 3 data pairs).
I haven't run this model for more than a couple of iterations.  

I haven't used any evaluation metric (like BLEU) either, because there's not much point in evaluating these low quality predictions. These are on my TO-DO list if I later find oppurtunity for proper training and testing.

I used valina RNN as a base for simplicity (and because with intra-layer attention, long range dependencies can be included even in an otherwise vanilla RNN), but LSTM or GRU is probably a better choice to avoid gradient vanishing\exploding. 

#### Interesting paper ( which I recently discovered ) dealing with related matters: 

[A Deep Reinforced Model For Abstractive Summarization - Romain Paulus, Caiming Xiong & Richard Socher](https://arxiv.org/pdf/1705.04304.pdf)

The LSTMN model proposed in the paper below, is based on exactly similar intuitions which I had while building this model: 

[Long Short-Term Memory-Networks for Machine Reading - Jianpeng Cheng, Li Dong and Mirella Lapata](https://arxiv.org/pdf/1601.06733.pdf) 

