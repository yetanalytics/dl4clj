(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/MultiLayerConfiguration.html"}
    dl4clj.nn.api.multi-layer-conf
  (:import [org.deeplearning4j.nn.conf MultiLayerConfiguration])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code? eval-if-code]]))

(defn to-json
  [multi-layer-conf & {:keys [as-code?]
                       :or {as-code? true}}]
  (match [multi-layer-conf]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.toJson ~multi-layer-conf))
         :else
         (.toJson multi-layer-conf)))

(defn to-yaml
  [multi-layer-conf & {:keys [as-code?]
                       :or {as-code? true}}]
  (match [multi-layer-conf]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.toYaml ~multi-layer-conf))
         :else
         (.toYaml multi-layer-conf)))

(defn to-str
  [multi-layer-conf & {:keys [as-code?]
                       :or {as-code? true}}]
  (match [multi-layer-conf]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.toString ~multi-layer-conf))
         :else
         (.toString multi-layer-conf)))

(defn from-json
  [json & {:keys [as-code?]
           :or {as-code? true}}]
  (match [json]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(MultiLayerConfiguration/fromJson ~json))
         :else
         (MultiLayerConfiguration/fromJson json)))

(defn from-yaml
  [yaml & {:keys [as-code?]
           :or {as-code? true}}]
  (match [yaml]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(MultiLayerConfiguration/fromYaml ~yaml))
         :else
         (MultiLayerConfiguration/fromYaml yaml)))

(defn get-conf
  ":layer-idx (int), the index of the layer, within the multi-layer-conf, you
   want the configuration of"
  [& {:keys [multi-layer-conf layer-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:multi-layer-conf (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getConf ~multi-layer-conf (int ~layer-idx)))
         :else
         (let [[conf l-idx] (eval-if-code [multi-layer-conf seq?]
                                          [layer-idx seq? number?])]
           (.getConf conf l-idx))))

(defn get-input-pre-process
  ":layer-idx (int), the index of the layer, within the multi-layer-conf,
  that has a pre-processor."
  [& {:keys [multi-layer-conf layer-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:multi-layer-conf (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getInputPreProcess ~multi-layer-conf ~layer-idx))
         :else
         (let [[l-idx] (eval-if-code [layer-idx seq? number?])]
           (.getInputPreProcess multi-layer-conf l-idx))))
