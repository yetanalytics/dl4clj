(ns dl4clj.nn.api.updater
  (:import [org.deeplearning4j.nn.api Updater])
  (:require [dl4clj.nn.conf.utils :refer [contains-many?]]))

(defn clone
  [this]
  (.clone this))

(defn get-state-view-array
  [this]
  (.getStateViewArray this))

(defn set-state-view-array
  "Set the internal (historical) state view array for this updater"
  [& {:keys [this layer view-array initialize?]}]
  (.setStateViewArray this layer view-array initialize?))

(defn state-size-for-layer
  "Calculate and return the state size for this updater (for the given layer)."
  [& {:keys [this layer]}]
  (.stateSizeForLayer this layer))

(defn update-updater
  "Updater: updates the model"
  [& {:keys [this layer gradient iteration mini-batch-size]}]
  (.update this layer gradient iteration mini-batch-size))
