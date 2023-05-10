import argparse
import os


def test_commit(message=None, dry_run=False): #pragma: no cover

    run_tests_command = "coverage run -m tests.test_all"
    make_report_command = "coverage report > temp_coverage.txt"

    print(f"running on command line: \n  {run_tests_command}")
    os.system(run_tests_command)
    print(f"running on command line: \n  {make_report_command}")
    os.system(make_report_command)

    with open("temp_coverage.txt", "r") as f:
        for line in f.readlines():
            if "TOTAL" in line:
                summary = line

                while "\n" in summary:
                    idx = summary.find("\n")
                    summary = summary[:idx] + summary[idx+1:]

    git_add_command = "git add coverage.txt README.md"
    commit_command = f"git commit -m 'test commit summary: {summary}' "

    if message is not None:
        commit_command += f"-m '{message}'"

    if dry_run: 
        print("dry run, not running these commands:")
        print(git_add_command)
        print(commit_command)
        cleanup_command = "rm temp_coverage.txt"
        os.system(cleanup_command)
    else:
        readme_lines = []
        with open("README.md", "r") as f:
            for line in f.readlines():
                if "TOTAL" in line:
                    readme_lines.append(f"{summary} [coverage.txt](coverage.txt)")
                else: 
                    readme_lines.append(line)

        with open("README.md", "w") as f:

            f.writelines(readme_lines)

        os.system("mv temp_coverage.txt coverage.txt")
        os.system(git_add_command)
        os.system(commit_command)


if __name__ == "__main__": #pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--message", type=str, default=None,\
            help="optional, additional message to add to commit")
    parser.add_argument("-d", "--dry_run", dest="dry_run", action="store_true",\
            help="optional, run tests, but don't commit")
    parser.set_defaults(dry_run=False)

    args = parser.parse_args()

    test_commit(message=args.message, dry_run=args.dry_run)

