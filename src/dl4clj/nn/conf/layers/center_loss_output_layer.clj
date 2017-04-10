(ns dl4clj.nn.conf.layers.center-loss-output-layer
  (:import [org.deeplearning4j.nn.conf.layers CenterLossOutputLayer])
  (:require [dl4clj.nn.conf.layers.shared-fns :refer :all]))

(defn get-alpha [& {:keys [this]}]
  (.getAlpha this))

(defn get-gradient-check [& {:keys [this]}]
  (.getGradientCheck this))

(defn get-lambda [& {:keys [this]}]
  (.getLambda this))

;; other layer interaction fns can be found in the shared-fns ns
