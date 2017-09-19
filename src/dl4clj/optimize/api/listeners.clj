(ns dl4clj.optimize.api.listeners
  (:import [org.deeplearning4j.optimize.api IterationListener])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code?]]
            [clojure.java.io :refer [as-file]]))

(defn invoked?
  "Was the listener invoked?"
  [& {:keys [listener as-code?]
      :or {as-code? true}}]
  (match [listener]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.invoked ~listener))
         :else
         (.invoked listener)))

(defn export-scores!
  "exports the scores from the collection scores listener
  to a file or output stream

  :file (str), a file to write to (its path)

  :delim (str), the delimiter for the file or output stream

  :output-stream (output stream), an output stream to write to

  returns the listener"
  [& {:keys [listener file delim output-stream as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:listener (_ :guard seq?)
           :file (:or (_ :guard string?)
                      (_ :guard seq?))
           :delim (:or (_ :guard string?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~listener (.exportScores (as-file ~file) ~delim)))
         [{:listener _
           :file _
           :delim _}]
         (doto listener (.exportScores (as-file file) delim))
         [{:listener (_ :guard seq?)
           :output-stream (_ :guard seq?)
           :delim (:or (_ :guard string?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~listener (.exportScores ~output-stream ~delim)))
         [{:listener _
           :output-stream _
           :delim _}]
         (doto listener (.exportScores output-stream delim))
         [{:listener (_ :guard seq?)
           :file (:or (_ :guard string?)
                      (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~listener (.exportScores (as-file ~file))))
         [{:listener _
           :file _}]
         (doto listener (.exportScores (as-file file)))
         [{:listener (_ :guard seq?)
           :output-stream (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~listener (.exportScores ~output-stream)))
         [{:listener _
           :output-stream _}]
         (doto listener (.exportScores output-stream))))

(defn get-scores-vs-iter
  "currently results in a stack over flow error,

  will need to look more into this but I have a feeling

  this is not a user facing method"
  [& {:keys [listener as-code?]
      :or {as-code? true}}]
  (match [listener]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getScoreVsIter ~listener))
         :else
         (.getScoreVsIter listener)))
