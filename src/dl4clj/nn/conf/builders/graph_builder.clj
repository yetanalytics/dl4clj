(ns dl4clj.nn.conf.builders.graph-builder
  (:require [dl4clj.nn.conf.constants :as constants]
            [dl4clj.nn.conf.input-pre-processor :as pre-process])
  (:import [org.deeplearning4j.nn.conf ComputationGraphConfiguration$GraphBuilder]))


(defn graph-builder
  [nn-conf-builder {:keys [add-inputs add-layer add-vertex backprop
                           backprop-type input-pre-processor pretrain
                           input-type set-outputs tbptt-back-length tbptt-fwd-length]
                    :or {}
                    :as opts}]
  (let [b (ComputationGraphConfiguration$GraphBuilder. nn-conf-builder)]
    (if (contains? opts :add-inputs)
      (.addInputs b add-inputs) b)
    (if (contains? opts :add-layer)
      (.addLayer b add-layer) b)
    (if (contains? opts :add-vertex)
      (.addVertex b add-vertex) b)
    (if (contains? opts :backprop)
      (.backprop b backprop) b)
    (if (contains? opts :backprop-type)
      (.backpropType b (constants/value-of {:backprop-type backprop-type})) b)
    (if (contains? opts :input-pre-processor)
      (.inputPreProcessor b (pre-process/pre-processors input-pre-processor)) b)
    (if (contains? opts :pretrain)
      (.pretrain b pretrain) b)
    (if (contains? opts :input-type)
      (.setInputTypes b input-type) b)
    (if (contains? opts :set-outputs)
      (.setOutputs b set-outputs) b)
    (if (contains? opts :tbptt-back-length)
      (.tBPTTBackwardLength b tbptt-back-length)
      b)
    (if (contains? opts :tbptt-fwd-length)
      (.tBPTTForwardLength b tbptt-fwd-length)
      b)
    b
    ))
