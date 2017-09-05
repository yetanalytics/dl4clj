(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/MultiLayerConfiguration.html"}
    dl4clj.nn.api.multi-layer-conf
  (:import [org.deeplearning4j.nn.conf MultiLayerConfiguration])
  (:require [clojure.core.match :refer [match]]))

(defn to-json
  [multi-layer-conf]
  (match [multi-layer-conf]
         [(_ :guard seq?)]
         `(.toJson ~multi-layer-conf)
         :else
         (.toJson multi-layer-conf)))

(defn to-yaml
  [multi-layer-conf]
  (match [multi-layer-conf]
         [(_ :guard seq?)]
         `(.toYaml ~multi-layer-conf)
         :else
         (.toYaml multi-layer-conf)))

(defn to-str
  [multi-layer-conf]
  (match [multi-layer-conf]
         [(_ :guard seq?)]
         `(.toString ~multi-layer-conf)
         :else
         (.toString multi-layer-conf)))

(defn from-json
  [json]
  (match [json]
         [(_ :guard seq?)]
         `(MultiLayerConfiguration/fromJson ~json)
         :else
         (MultiLayerConfiguration/fromJson json)))

(defn from-yaml
  [json]
  (match [json]
         [(_ :guard seq?)]
         `(MultiLayerConfiguration/fromYaml ~json)
         :else
         (MultiLayerConfiguration/fromYaml json)))

(defn get-conf
  ":layer-idx (int), the index of the layer, within the multi-layer-conf, you
   want the configuration of"
  [& {:keys [multi-layer-conf layer-idx]
      :as opts}]
  (match [opts]
         [{:multi-layer-conf (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(.getConf ~multi-layer-conf (int ~layer-idx))
         :else
         (.getConf multi-layer-conf layer-idx)))

(defn get-input-pre-process
  ":layer-idx (int), the index of the layer, within the multi-layer-conf,
  that has a pre-processor."
  [& {:keys [multi-layer-conf layer-idx]
      :as opts}]
  (match [opts]
         [{:multi-layer-conf (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(.getInputPreProcess ~multi-layer-conf ~layer-idx)
         :else
         (.getInputPreProcess multi-layer-conf layer-idx)))
