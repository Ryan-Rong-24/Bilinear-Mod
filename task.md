Train a bilinear layer on modular arithmatic, plotting some of the interaction matrices & top eigenvector components.



Modular arithmatic paper: I'd look at this LW post: https://www.lesswrong.com/posts/cbDEjnRheYn38Dpc5/interpreting-modular-addition-in-mlps

Note that it's mod P, where P=113 & inputs are one-hot vector.



Bilinear layer paper: https://arxiv.org/pdf/2410.08417?

Interaction Matrix Formula: Top of page 3

Top eigen-vectors of 3rd-order tensor: Section 3.3 



Note: The input to a bilinear layer is X, so you should have the same input in both encoder-matrices of the bilinear layer. Additionally, weight decay is required.



Don't spend more than 2 hours on this; just submit what you have. You are highly encouraged to use LLMs (I don't think you would complete this in under 2 hours on your own).  I highly recommend Claude Code in general, as well as copy-pasting the entire paper into your favorite LLM to ask questions.



Please submit results/images as a google doc. 



Feel free to do additional exploration or training on other tasks if time allows. 