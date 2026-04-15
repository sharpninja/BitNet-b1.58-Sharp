using System;

namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// Payload a worker POSTs to <c>/gradient</c> after it finishes training
/// on the task assigned to it. The coordinator uses the metadata to
/// compute staleness compensation and the payload itself to apply the
/// gradient to the global weight copy.
/// </summary>
/// <param name="TaskId">Task this submission satisfies.</param>
/// <param name="WorkerId">Worker that produced the gradient.</param>
/// <param name="BaseWeightVersion">Version the worker computed the
/// gradient against. The coordinator uses this to compute staleness =
/// (current_global_version - BaseWeightVersion).</param>
/// <param name="TokensSeen">Actual number of training tokens the worker
/// consumed. Fewer than the task's TokensPerTask is allowed (partial
/// progress) but the coordinator may reject tasks that fall below a
/// minimum fraction.</param>
/// <param name="LossAfter">Training loss the worker measured on its
/// local batch after the final local step.</param>
/// <param name="GradientFormat">Gradient encoding: "int8-ef" (int8 with
/// error feedback, v1 default), "fp32", "ternary-1.58", "topk-sparse".
/// The coordinator rejects submissions whose format it was not
/// configured to accept.</param>
/// <param name="GradientPayload">Opaque byte array containing the
/// encoded gradient. Layout is determined by <see cref="GradientFormat"/>
/// and is not interpreted at the Contracts layer.</param>
/// <param name="WallClockMs">How long the worker took to execute the
/// task end-to-end. Feeds into the coordinator's efficiency estimate.</param>
public sealed record GradientSubmission(
    string TaskId,
    string WorkerId,
    long BaseWeightVersion,
    long TokensSeen,
    double LossAfter,
    string GradientFormat,
    byte[] GradientPayload,
    long WallClockMs);
