(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/Updater.html"}
  dl4clj.nn.conf.updater
  (:import [org.deeplearning4j.nn.conf Updater])
  (:require [clojure.string :as s]))

(defn value-of [k]
  (if (string? k)
    (Updater/valueOf k)
    (Updater/valueOf (s/replace (s/upper-case (name k)) "-" "_"))))

(defn values
  "Returns collection of supported updaters:
  (:sgd :adam :adadelta :nesterovs :adagrad :rmsprop)
  "
  []
  (map #(keyword (s/replace (s/lower-case (str %)) "_" "-")) (Updater/values)))

(comment

  (map value-of (values))
  (value-of :adagrad)
  (value-of :sgd)
  (value-of :adam)
  (value-of :adadelta)
  (value-of :nesterovs)
  (value-of :adagrad)
  (value-of :rmsprop)
  (value-of :none)
  (value-of :custom)
  (value-of "ADAGRAD")

)
