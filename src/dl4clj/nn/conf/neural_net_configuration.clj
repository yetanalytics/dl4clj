(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/NeuralNetConfiguration.html"}
  dl4clj.nn.conf.neural-net-configuration
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration]))

(defn add-variable [this s]
  (.addVariable this s))

(defn clear-variables [this]
  (.clearVariables this))

(defn clone [this]
  (.clone this))

(defn from-json [json]
  (.fromJson json))

(defn from-yaml [json]
  (.fromYaml json))

(defn get-l1-by-param [this s]
  (.getL1ByParam this s))

(defn get-l2-by-param [this s]
  (.getL2ByParam this s))

(defn get-learning-rate-by-param [this s]
  (.getLearningRateByParam this s))

(defn mapper [this]
  (.mapper this))

(defn mapper-yaml [this]
  (.mapperYaml this))

(defn reinit-mapper-with-subtypes [typez]
  (.reinitMapperWithSubtypes typez))

(defn set-layer-param-lr [this s]
  (.setLayerParamLR this s))

(defn set-learning-rate-by-param [this s rate]
  (.setLearningRateByParam this s rate))

(defn list-variables
  ([this]
   (.variables this))
  ([this t-or-f]
   (.variables this t-or-f)))

(defn to-json [cfg]
  (.toJson cfg))

(defn build-nn [this]
  (.build this))
