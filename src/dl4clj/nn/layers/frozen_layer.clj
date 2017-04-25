(ns ^{:doc "For purposes of transfer learning A frozen layers wraps another dl4j layer within it.
 The params of the layer within it are frozen or in other words held constant.
During the forward pass the frozen layer behaves as the layer within it would during test
regardless of the training/test mode the network is in. Backprop is skipped since parameters are not be updated.
Implementation of the class FrozenLayer in dl4j.
all refered fns support frozen-layer as the first arg
see https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/FrozenLayer.html"}
    dl4clj.nn.layers.frozen-layer
  (:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.conf.constants :as enum])
  (:import [org.deeplearning4j.nn.layers FrozenLayer]))

(defn new-frozen-layer
  "creates a new frozen layer from an existing layer"
  [layer-to-be-frozen]
  (FrozenLayer. layer-to-be-frozen))

(defn log-test-mode
  ;; figure out exactly what this does. no docs for this method
  ":training? (boolean) training or testing?
  :training-mode (keyword) one of :training or :testing"
  [& {:keys [frozen-layer training? training-mode]
      :as opts}]
  (assert (contains? opts :frozen-layer) "you must supply a frozen layer")
  (cond (contains? opts :training?)
        (doto frozen-layer
          (.logTestMode training?))
        (contains? opts :training-mode)
        (doto frozen-layer
          (.logTestMode (enum/value-of {:layer-training-mode training-mode})))
        :else
        (assert false "you must supply training? or training-mode")))

(defn get-inside-layer
  [frozen-layer]
  (.getInsideLayer frozen-layer))
