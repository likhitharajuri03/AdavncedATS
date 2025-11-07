# Smoke test for finalATS (PowerShell)
# Usage: .\scripts\smoke_test.ps1 [-HostUrl <string>]
# Default HostUrl is http://localhost:8080 (nginx). If you expose backend directly, use http://localhost:5000
param(
    [string]$HostUrl = "http://localhost:8080"
)

Write-Host "Running smoke test against $HostUrl ..."

function Call-Api($method, $url, $body = $null, $token = $null) {
    $headers = @{}
    if ($token) { $headers.Add('Authorization', "Bearer $token") }
    if ($body -ne $null) {
        return Invoke-RestMethod -Uri $url -Method $method -Body ($body | ConvertTo-Json -Depth 8) -ContentType 'application/json' -Headers $headers -ErrorAction Stop
    } else {
        return Invoke-RestMethod -Uri $url -Method $method -Headers $headers -ErrorAction Stop
    }
}

# 1) Health check
try {
    $health = Call-Api -method 'GET' -url "$HostUrl/api/health"
    Write-Host "Health check OK:" (ConvertTo-Json $health -Depth 3)
} catch {
    Write-Error "Health check failed: $_"
    exit 1
}

# 2) Register a test user
$testUser = @{ email = "smoke_test_user@example.com"; password = "Sm0keTest!"; name = "Smoke Tester"; user_type = "job_seeker" }
try {
    $reg = Call-Api -method 'POST' -url "$HostUrl/api/auth/register" -body $testUser
    $token = $reg.access_token
    Write-Host "Registered user and obtained token."
} catch {
    Write-Warning "Register may have failed (user may already exist). Trying login..."
    try {
        $loginResp = Call-Api -method 'POST' -url "$HostUrl/api/auth/login" -body @{ email = $testUser.email; password = $testUser.password }
        $token = $loginResp.access_token
        Write-Host "Login succeeded. Obtained token."
    } catch {
        Write-Error "Register & login both failed: $_"
        exit 1
    }
}

# 3) Run a simple resume analysis
$payload = @{
    resume_text = "John Doe\nExperienced software engineer with 5 years of backend development in Python, Flask, and PostgreSQL. Familiar with AWS and containerization.";
    job_description = "Looking for a backend engineer experienced in Python, Flask, PostgreSQL, and AWS."
}
try {
    $analysis = Call-Api -method 'POST' -url "$HostUrl/api/resume/analyze" -body $payload -token $token
    Write-Host "Resume analysis result:" (ConvertTo-Json $analysis -Depth 6)
} catch {
    Write-Error "Resume analysis failed: $_"
    exit 1
}

Write-Host "Smoke test completed successfully." -ForegroundColor Green
