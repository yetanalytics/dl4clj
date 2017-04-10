(ns dl4clj.nn.conf.layers.base-output-layer
  (:import [org.deeplearning4j.nn.conf.layers BaseOutputLayer])
  (:require [dl4clj.nn.conf.layers.shared-fns :refer :all]))

(defn get-loss-fn [& {:keys [this]}]
 (.getLossFn this))
