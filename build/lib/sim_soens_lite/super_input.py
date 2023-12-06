import numpy as np


from .soen_sim import input_signal
from .super_functions import *

class SuperInput():
    '''
    Input object class
     - generates an input object that is compatible with any soen_sim synapse
     - by default, takes the form of an arbitrary spike train or n>=1 channels
     - offers:
        - random input (channels, total_spikes, duration)
        - defined input (channels, spikes, duration)
        - MNIST
        - saccade MNIST
    '''
    def __init__(self,**entries):
        self.type = "random"
        self.duration = 500
        self.channels = 10
        self.slow_down = 10
        self.total_spikes = 25
        self.sample = 0
        self.name = 'SuperInput'
        self.__dict__.update(entries)
        self.temporal_form = 'arbitrary_spike_train'
        
        if self.type == "random":
            self.random()

        elif self.type == "defined":
            self.defined()
        
        elif self.type == "MNIST":
            self.MNIST()

        elif self.type == "saccade_MNIST":
            self.saccade_MNIST()

        elif self.type == "constant":
            self.constant()

        else:
            print("Please provide valid input type")
            pass
        

        self.signals = []
        for i in range(self.channels):
            array = np.sort(self.spike_rows[i])
            if array.any():
                array = np.append(array,np.max(array)+.001)
            self.signals.append(input_signal(name = 'input_synaptic_drive', 
                                input_temporal_form = self.temporal_form, 
                                spike_times = array) )
            # print(self.spike_rows[i])
        
    def random(self):
        '''
        SuperInput.random():
         - syntax:
           input = SuperInput(
            type         = 'random',  # specify input type
            channels     = 10,        # how many input channels
            total_spikes = 100,       # how many spikes in total
            duration     = 500        # over what duration
            )
         - use: generate a SuperInput object to be added as input to any synapse
        '''
        # print("Generating random input...")
        indices = np.random.randint(self.channels,size=self.total_spikes)
        times = np.random.rand(self.total_spikes)*self.duration
        self.spike_arrays = [indices,times]
        self.spike_rows = self.array_to_rows(self.spike_arrays)

    def defined(self):
        '''
        SuperInput.defined():
         - syntax:
           input = SuperInput(
            type         = 'defined',   # specify input type
            channels     = 10,          # how many input channels
            spike_times  = [27,54,...], # can be array of times only
                                        # or list of two arrays [indices, times]
            )
         - use: generate a SuperInput object to be added as input to any synapse
        '''
        # print("Generating defined input...")
        self.defined_spikes = np.array(self.defined_spikes)
        if self.defined_spikes.shape[0] != 2:
            # print("auto_times")
            indices = np.zeros(len(self.defined_spikes)).astype(int)
            self.defined_spikes = np.array([indices,self.defined_spikes])
        self.spike_arrays = self.defined_spikes
        self.spike_rows = self.array_to_rows(self.spike_arrays)
        
    def MNIST(self):
        '''
        input= SuperInput(
            type='MNIST',  # specify input type  
            index=i,       # specify input type MNIST digit
            sample=j,      # MNIST sample
            slow_down=100, # factor by which to slow down rate encoding
            duration=1000 # duration of input
            )
        '''
        self.channels = int(28*28)
        # print("Generating MNNIST dataset...")
        mnist_indices, mnist_spikes = self.make_MNIST()
        self.spike_arrays = [mnist_indices,mnist_spikes]
        self.spike_rows = self.array_to_rows(self.spike_arrays)

    def saccade_MNIST(self):
        '''
        SuperInput.defined():
         - syntax:
           input = SuperInput(
            type      = "saccade_MNIST", # specify input type
            channels  = 36,              # n*n pixels tile (must be a square)
            tile_time = 50,              # how long to rate encode each tile
            )
         - use: generate a SuperInput object to be added as input to any synapse
        '''
        self.channels = 36
        self.spike_arrays = self.make_saccade_MNIST()
        self.spike_rows = self.array_to_rows(self.spike_arrays)

    def constant(self):
        input = input_signal(name = 'constant_input', 
                             input_temporal_form = 'constant', 
                             applied_flux = self.phi_app)
        return input 

    def gen_ordered_input(self):
        pass

    def load_input(self):
        pass

    def rows_to_array(self,input):
        spikes = [ [] for _ in range(self.channels) ]
        count = 0
        for n in range(self.channels):
            if np.any(input[n]):
                # print(n)
                spikes[0].append(np.ones(len(input[n]))*n)
                spikes[1].append(input[n])
            count+=1
        spikes[0] =np.concatenate(spikes[0])
        spikes[1] = np.concatenate(spikes[1])
        return spikes

    def array_to_rows(self,array):
        rows = [ [] for _ in range(self.channels) ]
        for i in range(len(array[0])):
            rows[int(array[0][i])].append(array[1][i])
        return rows

    def make_MNIST(self):
        import brian2
        from keras.datasets import mnist
        # print("load")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # print("loaded")
        # simplified classification (0 1 and 8)
        # X_train = X_train[(y_train == 0) | (y_train == 1) | (y_train == 2)]
        # y = y_train[(y_train == 0) | (y_train == 1) | (y_train == 2)]

        # pixel intensity to Hz (255 becoms ~63Hz)
        # X_train = X_train / 4 
        numbers = [0,1,2]
        classes = ["zero", "one", "two"]

        # Generate spiking data
        brian2.start_scope()
        self.units = []
        self.times = []
        self.data = {}
        channels = 28*28
        # X = X_train[self.index].reshape(channels)

        X = (X_train[(y_train == self.index)][self.sample]).reshape(channels)

        # print(y_train[self.index])
        P = brian2.PoissonGroup(channels, rates=(X/self.slow_down)*brian2.Hz)
        MP = brian2.SpikeMonitor(P)
        net = brian2.Network(P, MP)
        net.run(self.duration*brian2.ms)
        spikes_i = np.array(MP.i[:])
        spikes_t = np.array(MP.t[:])
        self.indices = spikes_i
        self.times = spikes_t*1000

        return self.indices, self.times

    def make_saccade_MNIST(self):
        import matplotlib.pyplot as plt
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = X_train[(y_train == 0) | (y_train == 1) | (y_train == 2)]
        y = y_train[(y_train == 0) | (y_train == 1) | (y_train == 2)]
        
        dataset = [[] for i in range(3)]
        stream = [[],[]]
        count = 0
        for i in range(20):
            if len(dataset[y[i]]) < 3:
                dataset[y[i]].append(aug_digit(X[i]))

        for data in dataset:
            for sample in data:
                tiles = tile_img(sample)
                tile_spikes = tiles_to_spikes(tiles,self.tile_time)
                stream[0].extend(tile_spikes[0])
                stream[1].extend(np.array(tile_spikes[1])+(self.tile_time*36)*count)
                count+=1

        return stream
    
    def plot(self,**kwargs):
        from sim_soens_lite.soen_plotting import raster_plot
        raster_plot(self.spike_arrays,**kwargs)

