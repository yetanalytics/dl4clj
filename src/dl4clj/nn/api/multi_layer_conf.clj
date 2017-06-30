(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/MultiLayerConfiguration.html"}
    dl4clj.nn.api.multi-layer-conf
  (:import [org.deeplearning4j.nn.conf MultiLayerConfiguration]))

(defn to-json
  [multi-layer-conf]
  (.toJson multi-layer-conf))

(defn to-yaml
  [multi-layer-conf]
  (.toYaml multi-layer-conf))

(defn to-str
  [multi-layer-conf]
  (.toString multi-layer-conf))

(defn from-json
  [json]
  (MultiLayerConfiguration/fromJson json))

(defn from-yaml
  [json]
  (MultiLayerConfiguration/fromYaml json))

(defn get-conf
  ":layer-idx (int), the index of the layer, within the multi-layer-conf, you
   want the configuration of"
  [& {:keys [multi-layer-conf layer-idx]}]
  (.getConf multi-layer-conf layer-idx))

(defn get-input-pre-process
  ":layer-idx (int), the index of the layer, within the multi-layer-conf,
  that has a pre-processor."
  [& {:keys [multi-layer-conf layer-idx]}]
  (.getInputPreProcess multi-layer-conf layer-idx))
