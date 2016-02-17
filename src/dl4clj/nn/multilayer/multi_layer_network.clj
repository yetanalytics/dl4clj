(ns ^{:doc "See http://deeplearning4j.org/doc/org/deeplearning4j/nn/multilayer/MultiLayerNetwork.html"} 
  dl4clj.nn.multilayer.multi-layer-network
  (:require [dl4clj.nn.conf.multi-layer-configuration :as ml-cfg]
            [dl4clj.nn.conf.neural-net-configuration :as nn-cfg]
            [dl4clj.nn.api.model :as model]
            [dl4clj.nn.api.classifier :as classifier])
  (:import [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.conf MultiLayerConfiguration MultiLayerConfiguration$Builder]))

(defn multi-layer-network [opts]
  (if (instance? MultiLayerConfiguration opts)
    (MultiLayerNetwork. opts)
    (MultiLayerNetwork. (.build ^MultiLayerConfiguration$Builder (ml-cfg/builder (update-in opts [:confs] #(map nn-cfg/neural-net-configuration %)))))))

(defn init [^MultiLayerNetwork mln]
  (.init mln))

(defn rnn-clear-previous-state 
  "Clear the previous state of the RNN layers (if any)."
  [^MultiLayerNetwork rnn]
  (.rnnClearPreviousState rnn))

(defn rnn-time-step 
  "If this MultiLayerNetwork contains one or more RNN layers: conduct forward pass (prediction) but using previous stored state for any RNN layers. The activations for the final step are also stored in the RNN layers for use next time rnnTimeStep() is called.
This method can be used to generate output one or more steps at a time instead of always having to do forward pass from t=0. Example uses are for streaming data, and for generating samples from network output one step at a time (where samples are then fed back into the network as input)
If no previous state is present in RNN layers (i.e., initially or after calling rnnClearPreviousState()), the default initialization (usually 0) is used.
Supports mini-batch (i.e., multiple predictions/forward pass in parallel) as well as for single examples.
Parameters:
input - Input to network. May be for one or multiple time steps. For single time step: input has shape [miniBatchSize,inputSize] or [miniBatchSize,inputSize,1]. miniBatchSize=1 for single example.
For multiple time steps: [miniBatchSize,inputSize,inputTimeSeriesLength]
Returns:
Output activations. If output is RNN layer (such as RnnOutputLayer): if input has shape [miniBatchSize,inputSize] i.e., is 2d, output has shape [miniBatchSize,outputSize] (i.e., also 2d).
Otherwise output is 3d [miniBatchSize,outputSize,inputTimeSeriesLength] when using RnnOutputLayer."
  [^MultiLayerNetwork rnn input]
  (.rnnTimeStep rnn input))

(defn get-layers [^MultiLayerNetwork mln]
  (into [] (.getLayers mln)))

(defn get-layer [^MultiLayerNetwork mln i]
  (.getLayer mln i))


;;; Model Interface

(defmethod model/fit MultiLayerNetwork 
  ([^MultiLayerNetwork net]
   (.fit net))
  ([^MultiLayerNetwork net x]
   (.fit net x)))

(defmethod model/num-params MultiLayerNetwork 
  ([^MultiLayerNetwork net]
   (.numParams net))
  ([^MultiLayerNetwork net x]
   (.numParams net x)))

;;; Classifier Interface

(defmethod classifier/fit MultiLayerNetwork 
  ([^MultiLayerNetwork net]
   (.fit net))
  ([^MultiLayerNetwork net x]
   (.fit net x)))


;;; Model interface (see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Model.html):
;;;   fit()
;;;   fit(INDArray data)

;;; Classifier interface (see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Classifier.html)
;;;   fit(DataSet data)
;;;   fit(DataSetIterator iter)
;;;   fit(INDArray examples, INDArray labels)
;;;   fit(INDArray examples, int[] labels)

