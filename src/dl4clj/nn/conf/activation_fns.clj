(ns dl4clj.nn.conf.activation-fns
  (:require [dl4clj.utils :refer [camelize camel-to-dashed]]
            [clojure.string :as s])
  (:import [org.nd4j.linalg.activations Activation]))

(defn value-of [k]
  (cond (string? k)
        (cond (s/includes? k "-")
              (Activation/valueOf (s/upper-case (s/join (s/split k #"-"))))
              :else
              (Activation/valueOf (s/upper-case k)))
        :else
        (value-of (name k))))

(defn values []
  (map #(keyword (camel-to-dashed (.name ^Activation %))) (Activation/values)))


(comment
  (map value-of (values))
  (value-of :cube)
  (value-of :elu)
  (value-of :hard-sigmoid)
  (value-of :hardsigmoid)
  (value-of :hard-tanh)
  (value-of :hardtanh)
  (value-of :identity)
  (value-of :leaky-relu)
  (value-of :leakyrelu)
  (value-of :relu)
  (value-of :rrelu)
  (value-of :r-relu)
  (value-of :sigmoid)
  (value-of :soft-max)
  (value-of :softmax)
  (value-of :soft-plus)
  (value-of :softplus)
  (value-of :soft-sign)
  (value-of :softsign)
  (value-of :tanh))
