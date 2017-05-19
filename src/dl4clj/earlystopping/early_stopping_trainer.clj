(ns ^{:doc "Class for conducting early stopping training locally

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/trainer/EarlyStoppingTrainer.html"}
    dl4clj.earlystopping.early-stopping-trainer
  (:import [org.deeplearning4j.earlystopping.trainer
            EarlyStoppingTrainer BaseEarlyStoppingTrainer])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn new-early-stopping-trainer
  "onducting early stopping training locally (single machine), for training a MultiLayerNetwork.

  :early-stopping-conf (early-stopping-configuration),
   - see: dl4clj.earlystopping.early-stopping-config

  :mln (model or conf), a build multilayer network or the configuration for a multilayer network
   - see: dl4clj.nn.conf.builders.multi-layer-builders

  :training-dataset-iterator (dataset-iterator), an iterator for a dataset where training? is set to true
   - see: dl4clj.datasets.datavec

  :listener (listener), a listener that implements the EarlyStoppingListener interface
   - when more work is done on listeners this doc will be updated"
  [& {:keys [early-stopping-conf mln training-dataset-iterator]
      :as opts}]
  (assert (contains-many? opts :early-stopping-conf :mln :training-dataset-iterator)
          "you must supply a configuration, multi-layer-network and a dataset-iterator")
  (EarlyStoppingTrainer. early-stopping-conf mln training-dataset-iterator))
