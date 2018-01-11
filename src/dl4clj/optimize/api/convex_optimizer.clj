(ns ^{:doc "implementation of the Convex Optimizer interface
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/ConvexOptimizer.html"}
    dl4clj.optimize.api.convex-optimizer
  (:import [org.deeplearning4j.optimize.api ConvexOptimizer])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code? eval-if-code]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; getters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-batch-size
  "returns the batch size for the optimizer"
  [optimizer & {:keys [as-code?]
                :or {as-code? true}}]
  (match [optimizer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.batchSize ~optimizer))
         :else
         (.batchSize optimizer)))

(defn get-conf
  "get the nn-conf associated with this optimizer"
  [optimizer & {:keys [as-code?]
                :or {as-code? true}}]
  (match [optimizer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getConf ~optimizer))
         :else
         (.getConf optimizer)))

(defn get-updater
  "returns the updater associated with this optimizer"
  [optimizer & {:keys [as-code?]
                :or {as-code? true}}]
  (match [optimizer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getUpdater ~optimizer))
         :else
         (.getUpdater optimizer)))

(defn get-gradient-and-score
  "the gradient and score for this optimizer"
  [optimizer & {:keys [as-code?]
                :or {as-code? true}}]
  (match [optimizer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.gradientAndScore ~optimizer))
         :else
         (.gradientAndScore optimizer)))

(defn get-score
  "The score for the optimizer so far"
  [optimizer & {:keys [as-code?]
                :or {as-code? true}}]
  (match [optimizer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.score ~optimizer))
         :else
         (.score optimizer)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; setters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn set-batch-size!
  "set the batch size for the optimizerizer

  :batch-size (int), the batch size

  returns the optimizerizer"
  [& {:keys [optimizer batch-size as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:optimizer (_ :guard seq?)
           :batch-size (:or (_ :guard number?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~optimizer (.setBatchSize (int ~batch-size))))
         :else
         (let [[o s] (eval-if-code [optimizer seq?]
                                   [batch-size seq? number?])]
           (doto o (.setBatchSize s)))))

(defn set-listeners!
  "sets the listeners for the supplied optimizerizer

  :listeners (collection), a collection of listeners
   - clojure data structures can be used

  returns the optimizerizer"
  [& {:keys [optimizer listeners as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:optimizer (_ :guard seq?)
           :listeners (:or (_ :guard coll?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~optimizer (.setListeners ~listeners)))
         :else
         (let [[o l] (eval-if-code [optimizer seq?] [listeners seq? vector?])]
           (doto o (.setListeners l)))))

(defn set-updater!
  "sets the updater for the optimizerizer

  :updater (updater), an updater to add to the optimizerizer

  returns the optimizerizer"
  [& {:keys [optimizer updater as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:optimizer (_ :guard seq?)
           :updater (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~optimizer (.setUpdater ~updater)))
         :else
         (let [[o u] (eval-if-code [optimizer seq?] [updater seq?])]
           (doto o (.setUpdater u)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; misc
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn optimizerize
  "calls optimizerize! returns a boolean"
  [& {:keys [optimizer as-code?]
      :or {as-code? true}}]
  (match [optimizer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.optimizerize ~optimizer))
         :else
         (.optimizerize optimizer)))
