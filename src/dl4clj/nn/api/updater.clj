(ns ^{:doc "fns from the dl4j Interface for updaters. Update the model.
see https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Updater.html"}
    dl4clj.nn.api.updater
  (:import [org.deeplearning4j.nn.api Updater])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn clone
  "create a copy of the updater"
  [updater]
  (.clone updater))

(defn get-state-view-array
  "returns an INDArray of the state-view of the updater"
  [updater]
  (.getStateViewArray updater))

(defn set-state-view-array!
  "Set the internal (historical) state view array for this updater.
  returns the updater

  :layer is a built layer from any-layer-builder
  :view-array is an INDArray of the state view of the updater
  :initialize? (boolean)"
  [& {:keys [updater layer view-array initialize?]}]
  (doto updater
    (.setStateViewArray layer view-array initialize?)))

(defn state-size-for-layer
  "Calculate and return the state size for this updater (for the given layer).
  :layer is a built layer from any-layer-builder"
  [& {:keys [updater layer]}]
  (.stateSizeForLayer updater layer))

(defn update-updater!
  "updates the model and returns the updater.

  :layer is a built layer from any-layer-bulder
  :gradient is a gradient (improve this desc)
  :iteration (int) number of iterations to perform
  :mini-batch-size (int) size of the mini-batch"
  [& {:keys [updater layer gradient iteration mini-batch-size]}]
  (doto updater
    (.update layer gradient iteration mini-batch-size)))
