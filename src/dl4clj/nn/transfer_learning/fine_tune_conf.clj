(ns ^{:doc "a class for fine tuning a nn conf.
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/transferlearning/FineTuneConfiguration.html
and
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/transferlearning/FineTuneConfiguration.Builder.html"}
    dl4clj.nn.transfer-learning.fine-tune-conf
  (:import [org.deeplearning4j.nn.transferlearning FineTuneConfiguration$Builder
            FineTuneConfiguration])
  (:require [dl4clj.constants :as enum]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; make the fine tuning conf
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-fine-tune-conf
  "creates a new fine tune configuration

  :activation-fn (keyword), the activation fn to change/add

  :n-iterations (int), the number of iterations to run

  :regularization? (boolean), should regularization be used?

  :seed (int or long), consistent randomization

  :build? (boolean), do you want to build the conf or not?
   - defaults to true"
  [& {:keys [activation-fn n-iterations regularization? seed build?]
    :or {build? true}
    :as opts}]
  (let [b (FineTuneConfiguration$Builder.)]
    (cond-> b
      (contains? opts :activation-fn)
      (.activation (enum/value-of {:activation-fn activation-fn}))
      (contains? opts :n-iterations)
      (.iterations n-iterations)
      (contains? opts :regularization?)
      (.regularization regularization?)
      (contains? opts :seed)
      (.seed seed)
      (true? build?)
      .build)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; apply the fine tune conf
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn applied-to-nn-conf!
  "applies a fine tune configuration to a supplied neural network configuration.

   Returns the mutated nn-conf"
  [& {:keys [fine-tune-conf nn-conf]}]
  (.appliedNeuralNetConfiguration fine-tune-conf nn-conf))

(defn nn-conf-from-fine-tune-conf
  "creates a neural network configuration builder from a fine tune configuration.

  the resulting nn-conf has the fine-tune-confs opts applied.

  :build? (boolean), determines if a nn-conf builder or nn-conf is returned"
  [& {:keys [fine-tune-conf build?]
      :or {build? false}}]
  (if (true? build?)
    (.build (.appliedNeuralNetConfigurationBuilder fine-tune-conf))
    (.appliedNeuralNetConfigurationBuilder fine-tune-conf)))
