(ns dl4clj.nn.layers.activation-layer
  (:import [org.deeplearning4j.nn.layers ActivationLayer])
  (:require [dl4clj.nn.api.layer :as l]
            [dl4clj.nn.conf.utils :as u]))

;; figure out how this is going to fit into the larger picture

(defn constructor
  [& {:keys [conf input]}]
  (if input
    (ActivationLayer. conf input)
    (ActivationLayer. conf)))

(defn activate
  [this training?]
  (l/activate :this this :training? training?))

(defn back-prop-gradient
  [this epsilon]
  (l/backprop-gradient :this this :epsilon epsilon))
