(ns dl4clj.optimize.api.listeners
  (:import [org.deeplearning4j.optimize.api IterationListener])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn invoked?
  "Was the listener invoked?"
  [listener]
  (.invoked listener))

(defn export-scores!
  "exports the scores from the collection scores listener
  to a file or output stream

  :file (java.io.File), a file to write to

  :delim (str), the delimiter for the file or output stream

  :output-stream (output stream), an output stream to write to

  returns the listener"
  [& {:keys [listener file delim output-stream]
      :as opts}]
  (cond (contains-many? opts :file :delim)
        (doto listener (.exportScores file delim))
        (contains-many? opts :output-stream :delim)
        (doto listener (.exportScores output-stream delim))
        (contains? opts :file)
        (doto listener (.exportScores file))
        (contains? opts :output-stream)
        (doto listener (.exportScores output-stream))
        :else
        (assert false "you must supply alteast a file or output stream to export to")))

(defn get-scores-vs-iter
  "currently results in a stack over flow error,

  will need to look more into this but I have a feeling

  this is not a user facing method"
  [listener]
  (.getScoreVsIter listener))
