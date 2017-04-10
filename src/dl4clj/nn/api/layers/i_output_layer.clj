(ns dl4clj.nn.api.layers.i-output-layer
  (:import [org.deeplearning4j.nn.api.layers IOutputLayer])
  (:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.classifier :refer :all]
            [dl4clj.nn.api.model :refer :all]))

(defn compute-score
  "Compute score after labels and input have been set."
  [& {:keys [this full-network-l1 full-network-l2 training?]}]
  (.computeScore this full-network-l1 full-network-l2 training?))

(defn compute-score-for-examples
  "Compute the score for each example individually, after labels and input have been set."
  [& {:keys [this full-network-l1 full-network-l2]}]
  (.computeScoreForExamples this full-network-l1 full-network-l2))

(defn get-labels
  "Get the labels array previously set with setLabels(INDArray)"
  [this]
  (.getLabels this))

(defn set-labels
  "Set the labels array for this output layer"
  [& {:keys [this labels]}]
  (.setLabels this labels))
