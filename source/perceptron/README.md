# Perceptron
Rosenblatt Perceptron artificial neuron

# Dependencies
1. python 3.6+     
2. numpy
3. matplotlib

# Runing 
use the below command:     
`python main.py <path to test file>`

# Code Descriptioni
`perceptron.py` has a implementation of a simple neuron.     
`main.py` is the driver class for perceptron class. it will     
parse data and feed test data to perceptron neuron. this    
script then calls `feedback` method of `perceptron.py` in     
its learning cycle.
after finishing learning, if it ever finishes, the learned     
weights would be evaluated by evaluation data. if there is     
no evaluation data this phase is skiped.     
if data is in 2d format (it means input data is a vector of     
 size 2) then input data and learned line which seperates     
two classes will be plotted.


