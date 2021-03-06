
# Multi-class weather classification in PyTorch
[Link to the original dataset](https://data.mendeley.com/datasets/4drtyfjtfy/1?ref=hackernoon.com)

## Requirements
Before running the scripts, all the necessary packages need to be installed

``` pip install -r requirements.txt ```


## How to get started

First of all, it is necessary to run the [script](https://github.com/maypink/WeatherDataset/blob/main/work_with_data/remove_script.py) to clear the data and delete pictures of an unsuitable format

Then run the [main](https://github.com/maypink/WeatherDataset/blob/main/main.py) file


## Graphs from TensorBoard

<table><tr>
<td> <img src="https://github.com/maypink/WeatherDataset/blob/main/images/Accuracy_test%20(1).svg" alt="Drawing" style="width: 350px;"/> </td>
<td> <img src="https://github.com/maypink/WeatherDataset/blob/main/images/Accuracy_train%20(1).svg" alt="Drawing" style="width: 350px;"/> </td>
</tr></table>


  Loss/test                                                                          |                                     Loss/train
------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------:
![Loss/test](https://github.com/maypink/WeatherDataset/blob/main/images/Loss_test%20(1).svg "Loss/test") |  ![Loss/train](https://github.com/maypink/WeatherDataset/blob/main/images/Loss_train%20(1).svg "Loss/train")

