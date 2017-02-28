(ns dl4clj.nn.conf.builders.graph-builder
  (:require [dl4clj.nn.conf.constants :as constants]
            [dl4clj.nn.conf.input-pre-processor :as pre-process])
  (:import [org.deeplearning4j.nn.conf ComputationGraphConfiguration$GraphBuilder
            ComputationGraphConfiguration
            NeuralNetConfiguration$Builder]))


(defn graph-builder
  "sets up and modifies a graph builder given a param map of options

  options are:

  :add-inputs (coll of strings) Specify the inputs to the network, and their associated labels.

  :add-layer {:layer-name (str) :layer {:layer-type params} or (layer-type params)
             :input-pre-processor (keyword) :layer-inputs (str)} adds in the preprocessor
             or
             {:layer-name (str) :layer {:layer-type params} or (layer-type params)
              :layer-inputs (str)} doesnt add in the preprocessor
  Add a layer and (optionally) an InputPreProcessor, with the specified name and specified inputs.

  :add-vertex ....

  :backprop (boolean) Whether to do back prop (standard supervised learning) or not

  :backprop-type (keyword) the type of backprop, one of :standard or :truncated-bptt

  :input-pre-processor {:layer-name string :input-pre-processor keyword}
  Specify the processors for a given layer These are used at each layer for doing
  things like normalization and shaping of input. Can be set within :add-layer

  :pretrain (boolean) Whether to do layerwise pre training or not

  :input-type (keywords) Specify the types of inputs to the network, so that:
   (a) preprocessors can be automatically added, and
   (b) the :n-ins (input size) for each layer can be automatically calculated and set
  The order here is the same order as :add-inputs.

  ...make a refrence to where input-type is used outside of this ns

  :set-outputs (coll of strings) Set the network output labels.
   -make sure you do (into-array ...)

  :tbptt-back-length (int) When doing truncated BPTT: how many steps of backward should we do?
  Only applicable when doing backpropType(BackpropType.TruncatedBPTT)

  :tbptt-fwd-length (int) When doing truncated BPTT: how many steps of forward pass
  should we do before doing (truncated) backprop? Only applicable when doing TruncatedBPTT
  Typically tBPTTForwardLength parameter is same as the tBPTTBackwardLength parameter,
  but may be larger than it in some circumstances (but never smaller)
  Ideally your training data time series length should be divisible by this"

  [nn-conf-builder {:keys [add-inputs add-layer add-vertex backprop
                           backprop-type input-pre-processor pretrain
                           input-type set-outputs tbptt-back-length tbptt-fwd-length]
                    :or {}
                    :as opts}]
  (let [b (ComputationGraphConfiguration$GraphBuilder. nn-conf-builder)]
    (if (contains? opts :add-inputs)
      (.addInputs b (into-array add-inputs)) b)
    (if (contains? opts :add-layer) ;; finish
      (let [{l-name :layer-name
             layer :layer
             pre-processor :input-pre-processor
             layer-inputs :layer-inputs} add-layer]
        (if (nil? pre-processor)
          ;; this is not done, add functionality to actualy get what params specify
          ;; add in the layer creation and pre-processor stuff
          (.addLayer b l-name layer (into-array layer-inputs))
          (.addLayer b l-name layer pre-processor (into-array layer-inputs))))
      b)
    ;;https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/graph/GraphVertex.html
    ;; going to need a vertexName (str), vertex ^, array of vertex inputs (strings)
    (if (contains? opts :add-vertex) ;;finish
      (.addVertex b add-vertex) b)
    (if (contains? opts :backprop)
      (.backprop b backprop) b)
    (if (contains? opts :backprop-type)
      (.backpropType b (constants/value-of {:backprop-type backprop-type})) b)
    (if (contains? opts :input-pre-processor) ;; finish
      (.inputPreProcessor b (pre-process/pre-processors {:unit-variance-processor "foo"} )) b)
    (if (contains? opts :pretrain)
      (.pretrain b pretrain) b)
    (if (contains? opts :input-type) ;; finish (into-array (constants/input-type)) i believe
      (.setInputTypes b input-type) b)
    (if (contains? opts :set-outputs)
      (.setOutputs b (into-array set-outputs)) b)
    (if (contains? opts :tbptt-back-length)
      (.tBPTTBackwardLength b tbptt-back-length)
      b)
    (if (contains? opts :tbptt-fwd-length)
      (.tBPTTForwardLength b tbptt-fwd-length)
      b)
    b))

(comment

 (.addLayer (ComputationGraphConfiguration$GraphBuilder. (NeuralNetConfiguration$Builder.))
           "foo" (dl4clj.nn.conf.builders.builders/dense-layer-builder
                  {:n-in 100
                   :n-out 1000
                   :layer-name "not foo"
                   :activation-fn "TANH"
                   :gradient-normalization :none})
            (into-array ["baz"])))
