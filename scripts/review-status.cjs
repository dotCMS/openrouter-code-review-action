const SUMMARY_MARKER = 'dotbot code review:';
const RUN_MARKER = '<!-- dotbot-review-run ';

module.exports = async function reportReviewStatus({ github, context, core, status }) {
  const pr = context.payload.pull_request;
  if (!pr) return;
  const { owner, repo } = context.repo;
  const run = {
    run_id: String(context.runId),
    attempt: process.env.GITHUB_RUN_ATTEMPT || '1',
    head_sha: pr.head.sha,
  };
  const current = await github.rest.pulls.get({ owner, repo, pull_number: pr.number });
  if (current.data.head.sha !== run.head_sha) {
    core.info('Review was superseded by a new PR head; leaving its summary untouched.');
    return;
  }
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner, repo, issue_number: pr.number, per_page: 100,
  });
  let summary = null;
  let newer = false;
  for (const comment of comments) {
    if (comment.user?.login !== 'github-actions[bot]' || !comment.body?.startsWith(SUMMARY_MARKER)) continue;
    const line = comment.body.split('\n')[1] || '';
    if (!line.startsWith(RUN_MARKER) || !line.endsWith(' -->')) continue;
    let previous;
    try {
      previous = JSON.parse(line.slice(RUN_MARKER.length, -4));
    } catch {
      core.warning('Ignoring invalid review-run metadata.');
      continue;
    }
    if (!previous || !/^\d+$/.test(previous.run_id) || !/^\d+$/.test(previous.attempt)) continue;
    if (previous.run_id === run.run_id && previous.attempt === run.attempt) summary = comment;
    if (BigInt(previous.run_id) > BigInt(run.run_id) ||
        (previous.run_id === run.run_id && BigInt(previous.attempt) > BigInt(run.attempt))) newer = true;
  }
  if (!summary && newer) {
    core.info('A newer review run exists; not creating an obsolete status comment.');
    return;
  }
  const headings = {
    in_progress: '⏳ Review in progress',
    failed: '❌ Review failed',
    cancelled: '⚠️ Review cancelled',
  };
  if (!headings[status]) throw new Error('Invalid review status');
  const server = process.env.GITHUB_SERVER_URL || 'https://github.com';
  const url = `${server}/${owner}/${repo}/actions/runs/${run.run_id}/attempts/${run.attempt}`;
  const body = [
    SUMMARY_MARKER,
    `${RUN_MARKER}${JSON.stringify(run)} -->`,
    '',
    `### ${headings[status]}`,
    'This attempt has no completed verdict. Earlier summaries do not represent this attempt.',
    'No automatic approval is granted by this status.',
    '',
    `Head: \`${run.head_sha}\``,
    `[Workflow run ${run.run_id} — attempt ${run.attempt}](${url})`,
  ].join('\n');
  if (summary) {
    await github.rest.issues.updateComment({ owner, repo, comment_id: summary.id, body });
  } else {
    await github.rest.issues.createComment({ owner, repo, issue_number: pr.number, body });
  }
};
