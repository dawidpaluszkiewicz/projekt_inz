# Article clustering tool

### About The Project
This tool is created as a BEng Thesis. It allows to partition set of articles into clusters according to the topic similarity.
Users can choose which part of the article the tool should take into consideration(title, abstract, keywords, article body).
Depending on the need articles can be split into most similar elements groups(not even distribution of data guaranteed) or enforce equal size clusters.
Articles may be in txt or pdf format.

### Prerequisites

* Python 3

### Installation

1. Clone the repo
```sh
git clone https://github.com/dawidpaluszkiewicz/projekt_inz
```
2. Install python packages
```sh
pip install requirements.txt
```

### Usage

This is how you can test if everything is properly setup

```sh
python main.py -o c -a e ../test_data 3 txt
```

It will generate a result file called kmeans_equal_size containing data about clustered articles

