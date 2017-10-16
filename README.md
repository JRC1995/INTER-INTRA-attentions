# Attention-Everywhere

This is a similar seq-2-seq model as [this](https://github.com/JRC1995/Abstractive-Summarization) ,used on [Amazon-Fine-Food-Dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews/data) for abstractive summarization
(and should be usable on translation data for machine translation). 

The main difference is in the encoder and decoder units. This model is inspired from the idea of [recurrent residual attention](https://arxiv.org/abs/1709.03714). 

The RRA model adds a weighted summation of K previous hidden states with the standard recurrent function. 
However, unlike the standard attention where the attention weights are computed from the context by a scoring function, here (in the RRA model), the weights are randomly initialized.

So, I implemented the same standard layer-wise attention mechanism function to calculate the attention weights for previous
hidden states. This can be said to be <b>intra-layer attention</b>.

Note: There may have been previous works on intra-layer-attention, but I didn't refer to any such work while making this implementation.

# Intra-layer-attention for Encoder

I used [global attention](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf) for the basic attention mechanism. 

At each time step of the encoder, a <i>candidate hidden state</i> is created by the standard RNN method (but with elu activation).

>candidate_hidden = elu(input * wxh + previous_hidden_state * whh + Bias)

At this point, all the previous hidden states are considered as analogous to the encoder hidden states for the layerwise-attention, and the candidate_hidden_state is considered as analogous to the current decoder hidden state for the layerwise attention. 

All the previous hidden state is then scored by the scoring function taking the current candidate hidden state into account.

The scoring function, I used is: Transpose(ht) * hs where ht = traget hidden, and hs = source hidden; in this context,
ht will be the current candidate hidden state, and hs will be the list of all previous hidden states.

Similar to the standard global attention mechanism, a context vector is created as a result of the weighted summation of
all the previous hidden states, and then an 'attended hidden state' is created by taking both the context vector and the candidate hidden state into account (the formula for creating ht_dash as given in [here](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf))

Next, I used the new attended hidden state as 'the hidden state' for computing the next hidden state. 

So we again put this in the standard RNN equation:

hidden_state = elu(input * wxh + attended_hidden_state * whh + Bias)

The normal attention mechanism can be said to 'inter-layer' attention mechanism (interaction between two different layers
(encoder, decoder) take place), whereas this type of attention mechanism can be said to be 'intra-layer' attention 
mechanism (interaction between previous and current hidden states take place within the same RNN layer) 


I used a similar process for the decoder.
First, the layer wise(or better 'inter-layer') attention mechanism calculates the inter-layer attended hidden state by the standard process of global-attention.
From the inter-layer attended hidden state the output token is computed.
Next, the 'intra-layer' attention mechanism starts its job: 

first it computes the candidate hidden state from the 
inter-layer attended hidden state AND the computed output token using the standard RNN formula as used in the encoder.

Second, similar to the encoders, it computes the intra-layer attended hidden state by the same process of scoring
previous hidden states and all else as done in the encoder.

Third, it computes 'THE decoder hidden state' from the intra-layer attended hidden state AND the output token
in the standard RNN formula as done in the encoder. 

Is it any better?

Can't say. For now, I don't have much (computational) resource for testing.

The code is somewhat ugly. Turned out, Tensorflow non-trainable Variable assignment don't work as I thought. 
I had to used tensorarrays. But tensorflow is kinda glitchy with tensorarray, especially dynamic ones. I faced
certain run-time errors, that I shouldn't. I noticed certain patterns, and took some convoluted steps (and some
'seemingly' redundent ones) to circumnavigate them. 

