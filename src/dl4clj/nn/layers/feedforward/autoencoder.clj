(ns ^{:doc "Autoencoder. Add Gaussian noise to input and learn a reconstruction function.
Implementation of the class AutoEncoder in dl4j.
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/feedforward/autoencoder/AutoEncoder.html"}
    dl4clj.nn.layers.feedforward.autoencoder
  (:import [org.deeplearning4j.nn.layers.feedforward.autoencoder AutoEncoder])
  (:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.layers.base-pretrain-network :refer [get-corrupted-input
                                                            sample-hidden-given-visible
                                                            sample-visible-given-hidden]]
            [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]]))

(defn new-autoencoder
  "creates a new convolution-1d-layer given a neural net conf and
  optionally some input data"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (AutoEncoder. conf input)
    (AutoEncoder. conf)))

(defn decode
  "decodes an encoded output of an autoencoder"
  [& {:keys [autoencoder layer-output]}]
  (.decode autoencoder layer-output))

(defn encode
  "encodes an input to be passed to an autoencoder"
  [& {:keys [autoencoder input training?]}]
  (.encode autoencoder input training?))
