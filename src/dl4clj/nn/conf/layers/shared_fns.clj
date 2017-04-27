(ns ^{:doc "see https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/Layer.html
and https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/package-frame.html"}
    dl4clj.nn.conf.layers.shared-fns
  (:import [org.deeplearning4j.nn.conf.layers Layer]))

;; should this be turned into a protocol for the other ns's to use?

;; top level layer interaction fns
;; from the Layer class
;; double check for when you need to use doto for methods that return nil
(defn clone [& {:keys [this]}]
  (.clone this))

(defn get-l1-by-param
  "Get the L1 coefficient for the given parameter."
  [& {:keys [this param-name]}]
  (.getL1ByParam this param-name))

(defn get-l2-by-param
  "Get the L2 coefficient for the given parameter."
  [& {:keys [this param-name]}]
  (.getL2ByParam this param-name))

(defn get-learning-rate-by-param
  "Get the (initial) learning rate coefficient for the given parameter."
  [& {:keys [this param-name]}]
  (.getLearningRateByParam this param-name))

(defn get-output-type [& {:keys [this layer-idx input-type]}]
  (.getOutputType this layer-idx input-type))

(defn get-pre-processor-for-input-type [& {:keys [this input-type]}]
  (.getPreProcessorForInputType this input-type))

(defn initializer [this]
  (.initializer this))

(defn instantiate [& {:keys [this conf listiner layer-idx
                             layer-param-view initialize-params?]}]
  (.instantiate this conf listiner layer-idx layer-param-view initialize-params?))

(defn set-n-in [& {:keys [this input-type override?]}]
  (.setNIn this input-type override?))

(defn reset-layer-default-config [& {:keys [this]}]
  (.resetLayerDefaultConfig this))

(defn get-updater-by-param [& {:keys [this param-name]}]
  (.getUpdaterByParam this param-name))
