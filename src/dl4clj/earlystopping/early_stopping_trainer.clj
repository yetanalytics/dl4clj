(ns ^{:doc "Class for conducting early stopping training locally

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/trainer/EarlyStoppingTrainer.html"}
    dl4clj.earlystopping.early-stopping-trainer
  (:import [org.deeplearning4j.earlystopping.trainer
            EarlyStoppingTrainer BaseEarlyStoppingTrainer])
  (:require [dl4clj.utils :refer [contains-many?]]
            [dl4clj.helpers :refer [reset-if-empty?!]]))

(defn new-early-stopping-trainer
  "onducting early stopping training locally (single machine), for training a MultiLayerNetwork.

  :early-stopping-conf (early-stopping-configuration),
   - see: dl4clj.earlystopping.early-stopping-config

  :mln (model or conf), a build multilayer network or the configuration for a multilayer network
   - see: dl4clj.nn.conf.builders.multi-layer-builders

  :iter (dataset-iterator), an iterator for a dataset where training? is set to true
   - see: dl4clj.datasets.datavec"
  [& {:keys [early-stopping-conf mln iter]
      :as opts}]
  (assert (contains-many? opts :early-stopping-conf :mln :iter)
          "you must supply a configuration, multi-layer-network and a dataset-iterator")
  (EarlyStoppingTrainer. early-stopping-conf mln (reset-if-empty?! iter)))
