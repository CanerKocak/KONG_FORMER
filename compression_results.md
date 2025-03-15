# Compression Results Summary

## Disk Size Comparison
- Original GPT-2 model: 541MB
- Compressed model: 531MB
- **Disk size reduction: 10MB (1.85%)**

## Parameter Count
- Original GPT-2 params: 144.99 million
- Compressed model params: 138.18 million
- **Parameter reduction: 6.81 million (4.7%)**

## Inference Speed (MPS)
- Original model avg inference time: 0.0397s
- Compressed model avg inference time: 0.0345s
- **Speedup factor: 1.14x (14% faster)**

## Text Generation Quality
The compressed model shows significant degradation in text generation quality:
- **Original model**: Produces coherent, multi-sentence outputs with reasonable context and flow
- **Compressed model**: Generates very short, often incomplete outputs with limited coherence

### Example Outputs

#### Prompt: "Artificial intelligence will"
- **Original**: "Artificial intelligence will take some time to build. Some of the biggest names in the field have already begun to invest in a large number of companies and will continue to do so, but many of the firms will need to build their own infrastructure..."
- **Compressed**: "Artificial intelligence will enable hardware and quantum computers for computation across quantum computing algorithms, and quantum computational algorithms and,"

#### Prompt: "The future of technology is"
- **Original**: "The future of technology is uncertain. 'The next generation of chips will bring new technologies to the market. However, we must always be vigilant about the risks and do not assume that the next generation will be better than the current one...'"
- **Compressed**: "The future of technology is 'upside-up.'"

## Conclusion
The compression implementation achieved:
- Minimal disk size reduction (1.85%)
- Small parameter count reduction (4.7%)
- Modest inference speed improvement (14%)
- Significant degradation in text generation quality

The compression appears to have been technically successful but at a severe cost to model quality. The compressed model generates much shorter, less coherent text compared to the original model, suggesting that important information was lost during compression.

## Recommendations
1. Adjust compression settings to be less aggressive
2. Try different compression layer placements
3. Increase the training time to allow the model to better adapt to compression
4. Consider alternative compression techniques that better preserve model quality 