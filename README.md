# Collection Characterizing Suite (CCS)
Repository for ACM HyperText 2018 paper [Data](/Data) and [Code](/Code)
## Citing Project
A publication related to this project appeared in the proceedings of ACM HyperText 2018 ([Read the PDF](https://www.cs.odu.edu/~mln/pubs/ht-2018/hypertext-2018-nwala-bootstrapping.pdf)). Please cite it as below:

> A. C. Nwala, M. C. Weigle, and M. L. Nelson, “Bootstrapping web archive collections
from social media,” in Proceedings of ACM Hypertext and Social Media (HT 2018),
pp. 64–72, 2018.

```latex
@inproceedings{ht-2018:nwala:ccs,
  author    = {Nwala, Alexander C. and Weigle, Michele C. and Nelson, Michael L.},
  title     = {{Bootstrapping Web Archive Collections from Social Media}},
  booktitle = {Proceedings of ACM Hypertext and Social Media (HT 2018)},
  series    = {HT '18},
  year      = {2018},
  month     = {jul},
  location  = {Baltimore, Maryland, USA},
  pages     = {64--72},
  numpages  = {9},
  url       = {https://doi.org/10.1145/3209542.3209560},
  doi       = {10.1145/3209542.3209560},
  isbn      = {9781450354271},
  publisher = {ACM},
  address   = {New York, NY, USA}
}
```
## CCS Metrics
Consider the following implementation details for the CCS metrics 
### Distribution of topics (Section 4.1)
Algorithm 1 which generates the distribution of topics has been further developed, optimized, and reimplemented as a Python application called [Sumgram](https://github.com/oduwsdl/sumgram/).

### Content diversity (Section 4.3)
Consider the following implementation [examples](/Code/contentDiversityExample.py) for measuring content diversity.

### Temporal distribution (Section 4.4)
[Newspaper](https://github.com/codelucas/newspaper) can be used to extract publication date from News articles when the date information is availble within the document. Otherwiser [CarbonDate](https://github.com/oduwsdl/CarbonDate) can be used to estimate the creation date of the document. SUTime can be used to extract relative datetime (e.g., "Next month") and normalize (e.g., 2020-09-01 if current month is August) them. SUTime is implemented in the Stanford CoreNLP suite ([installation option 1](https://ws-dl.blogspot.com/2018/03/2018-03-04-installing-stanford-corenlp.html), [installation option 2](https://stanfordnlp.github.io/CoreNLP/other-languages.html#docker))

### Source diversity (Section 4.5)
Consider the following blogpost about [measuring source diversity](https://github.com/anwala/url-diversity) and the [github](https://ws-dl.blogspot.com/2018/05/2018-05-04-exploration-of-url-diversity.html) that implements our source diversity method.

### Collection exposure (Section 4.6)
The archival state of a URI (`URI ARCHIVED` or `URI NOT ARCHIVED`) can be determined with [Sawood Alam's](https://twitter.com/ibnesayeed) [MemGator](https://github.com/oduwsdl/MemGator). The tweet index state of a URI (`URI IN TWEET` or `URI NOT IN TWEET`) can be determined by [searching for the URI](https://ws-dl.blogspot.com/2017/01/2017-01-23-finding-urls-on-twitter.html) on Twitter.

