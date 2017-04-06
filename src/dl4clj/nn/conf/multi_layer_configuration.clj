(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/MultiLayerConfiguration.html"}
    dl4clj.nn.conf.multi-layer-configuration
  (:import [org.deeplearning4j.nn.conf MultiLayerConfiguration]))


(defn to-json [^MultiLayerConfiguration cfg]
  (.toJson cfg))

(defn to-yaml [^MultiLayerConfiguration cfg]
  (.toYaml cfg))

(defn to-str [^MultiLayerConfiguration cfg]
  (.toString cfg))

(defn from-json [json]
  (.fromJson json))

(defn from-yaml [json]
  (.fromYaml json))

(defn clone [^MultiLayerConfiguration cfg]
  (.clone cfg))

(defn get-conf [^MultiLayerConfiguration cfg layer-n]
  (.getConf cfg (int layer-n)))

(defn get-input-pre-process [^MultiLayerConfiguration cfg layer-n]
  (.getInputPreProcess cfg layer-n))
